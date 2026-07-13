"""A pi0-style flow-matching action head on a Gemma 4 E2B backbone.

Ports the design LeRobot's pi0 uses (PaliGemma + action expert) onto Gemma 4, which is newer and
-- once the vocabulary is compacted (see model_utils) -- small enough to train on a 6GB GPU.

Why this and not the token-generation approach of v1-v4:

  actions      continuous, not 256-way bins
  generation   the whole 50-step chunk is denoised in parallel, not emitted token by token
  loss         MSE on the flow-matching velocity, not cross-entropy
  vision       the tower is trained, not frozen

The first three matter because greedy autoregressive decoding always takes the safest next token,
and when the future is ambiguous the safest answer is "don't move" -- which is exactly the flat
trajectory v1-v4 produced. A denoiser samples a trajectory instead of averaging over them. The
fourth matters because a frozen tower never learns what in the image decides where the arm goes;
pi0 ships with `freeze_vision_encoder=False` for the same reason.

Sequence fed to the LM as `inputs_embeds` (no tokenizer involved for the action part):

    [ image features | prompt text | state | noisy action chunk (50) ]
                                                    |
                                          last 50 hidden states -> velocity
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoConfig, BitsAndBytesConfig, Gemma4ForConditionalGeneration

from model_utils import EMB_KEY, MODEL, PLE_KEY, _local_model_dir

ACTION_DIM = 6
HORIZON = 50


def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the flow-matching timestep."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=-1)


# The vision tower's projections are wrapped in Gemma4ClippableLinear, which PEFT cannot adapt --
# that is why v1-v4 left the tower frozen. But the real nn.Linear sits one level in, as `.linear`,
# and PEFT does support Linear4bit, so LoRA can reach the vision encoder after all.
LORA_TARGETS = (
    r".*(language_model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|vision_tower\..*\.linear)$"
)


class Gemma4FlowPolicy(nn.Module):
    def __init__(self, base_ids: list[int]):
        super().__init__()
        local = _local_model_dir(base_ids, len(base_ids))  # vocab = just the prompt tokens
        cfg = AutoConfig.from_pretrained(local)
        self.cfg = cfg
        self.n_base = len(base_ids)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.vlm = Gemma4ForConditionalGeneration.from_pretrained(
            local, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}
        )
        # Hold the submodules directly: PEFT wraps `self.vlm` and shifts the attribute chain, but
        # it injects LoRA in place, so these references stay valid (and keep working).
        self.mm = self.vlm.model
        self.text_model = self.mm.language_model

        # Restore the rows of the two vocab tensors we kept (they are absent from the pruned
        # checkpoint); only the prompt tokens are ever embedded.
        from huggingface_hub import hf_hub_download

        wpath = hf_hub_download(MODEL, "model.safetensors")
        tm = self.text_model
        with safe_open(wpath, framework="pt", device="cpu") as f:
            emb = torch.stack([f.get_slice(EMB_KEY)[i : i + 1][0] for i in base_ids])
            ple = torch.stack([f.get_slice(PLE_KEY)[i : i + 1][0] for i in base_ids])
        tm.embed_tokens.weight.data = emb.to("cuda", torch.bfloat16)
        tm.embed_tokens_per_layer.weight.data = ple.to("cuda", torch.bfloat16)
        self.vlm.tie_weights()

        self.remap = torch.zeros(262144, dtype=torch.long, device="cuda")
        self.remap[torch.tensor(base_ids, device="cuda")] = torch.arange(self.n_base, device="cuda")

        h = cfg.text_config.hidden_size
        self.n_layers = cfg.text_config.num_hidden_layers
        self.hpl = cfg.text_config.hidden_size_per_layer_input

        # pi0's projections: state and the noisy action chunk enter the same stream as the image.
        self.state_proj = nn.Linear(ACTION_DIM, h)
        self.action_in_proj = nn.Linear(ACTION_DIM, h)
        self.action_time_mlp_in = nn.Linear(2 * h, h)
        self.action_time_mlp_out = nn.Linear(h, h)
        self.action_out_proj = nn.Linear(h, ACTION_DIM)
        for m in (self.state_proj, self.action_in_proj, self.action_time_mlp_in,
                  self.action_time_mlp_out, self.action_out_proj):
            m.to("cuda", torch.bfloat16)

        # Checkpoint both towers. The image alone expands to 2520 patches, and once the vision
        # tower is trainable that whole sequence has to be kept for the backward pass -- it OOMs
        # the GPU otherwise.
        #
        # `use_reentrant=False` is not optional here: the reentrant variant needs an input that
        # requires grad, and `pixel_values` does not, so the vision LoRA would silently receive no
        # gradients at all. `prepare_model_for_kbit_training()` is deliberately not used -- it
        # upcasts norms and embeddings to fp32, which then collides with the bf16 action head.
        gc_kwargs = {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        self.mm.vision_tower.gradient_checkpointing_enable(**gc_kwargs)
        self.vlm.gradient_checkpointing_enable(**gc_kwargs)
        self.vlm.config.use_cache = False

    def _prefix(self, pixel_values, image_position_ids, prompt_ids):
        """Image features followed by the prompt's text embeddings.

        `pooler_output` is the vision output already projected into the LM's embedding space --
        the tensor the model would normally scatter into the text sequence. We concatenate it
        instead, since we are assembling `inputs_embeds` ourselves.
        """
        vis = self.mm.get_image_features(pixel_values, image_position_ids=image_position_ids)
        img = vis.pooler_output[None].to(txt_dtype := self.text_model.embed_tokens.weight.dtype)
        txt = self.text_model.embed_tokens(prompt_ids).to(txt_dtype)
        return torch.cat([img, txt], dim=1)  # vision output is packed [n_tokens, H], batch 1

    def _suffix(self, state, x_t, t):
        h = self.cfg.text_config.hidden_size
        state_emb = self.state_proj(state)[:, None, :]  # [B,1,H]
        a = self.action_in_proj(x_t)  # [B,50,H]
        te = time_embedding(t, h).to(a.dtype)[:, None, :].expand_as(a)
        a = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(torch.cat([a, te], dim=-1))))
        return torch.cat([state_emb, a], dim=1)  # [B,51,H]

    def _block_mask(self, n_prefix: int, n_suffix: int, device, dtype):
        """pi0's block-attention mask, not the LM's default causal one.

        A denoiser has to see the *whole* noisy chunk to correct it; under causal masking each
        action step is denoised almost independently, which is what produced the high-frequency
        jitter in the first attempt. Blocks (following pi0's `make_att_2d_masks`):

            prefix (image + prompt)  one bidirectional block; cannot see the actions
            state                    sees the prefix and itself
            actions                  see everything, and each other, bidirectionally
        """
        s = n_prefix + n_suffix
        ar = torch.zeros(s, dtype=torch.long, device=device)
        ar[n_prefix] = 1  # state opens a block
        ar[n_prefix + 1] = 1  # the action chunk opens the next one
        cs = ar.cumsum(0)
        allow = cs[None, :] <= cs[:, None]  # i attends j iff j's block is not after i's
        mask = torch.zeros(1, 1, s, s, dtype=dtype, device=device)
        return mask.masked_fill(~allow, torch.finfo(dtype).min)

    def _backbone(self, prefix, suffix):
        """Run the LM over raw embeddings, supplying the per-layer inputs and mask ourselves.

        The PLE table is indexed by token id, but image/state/action positions have no token, so
        they reuse a single row -- which is what the model already does for image tokens anyway.
        Passing `attention_mask` as a dict is the documented escape hatch that stops the model from
        building its own causal mask.
        """
        embs = torch.cat([prefix, suffix], dim=1)
        b, s, _ = embs.shape
        tm = self.text_model
        ids = torch.zeros(b, s, dtype=torch.long, device=embs.device)
        ple = tm.embed_tokens_per_layer(ids).reshape(b, s, self.n_layers, self.hpl)
        mask = self._block_mask(prefix.shape[1], suffix.shape[1], embs.device, embs.dtype)
        out = tm(inputs_embeds=embs, per_layer_inputs=ple,
                 attention_mask={"full_attention": mask, "sliding_attention": mask})
        return out.last_hidden_state

    def loss(self, pixel_values, image_position_ids, prompt_ids, state, actions):
        """Flow matching: regress the velocity that carries noise onto the true action chunk."""
        b = actions.shape[0]
        noise = torch.randn_like(actions)
        # t must share the action dtype: a float32 t would promote x_t and then hit the bf16 head.
        t = torch.rand(b, device=actions.device, dtype=actions.dtype)
        x_t = t[:, None, None] * noise + (1 - t[:, None, None]) * actions
        u_t = noise - actions  # target velocity

        prefix = self._prefix(pixel_values, image_position_ids, prompt_ids)
        hid = self._backbone(prefix, self._suffix(state, x_t, t))[:, -HORIZON:]
        v_t = self.action_out_proj(hid.to(torch.bfloat16))
        return F.mse_loss(v_t.float(), u_t.float())

    @torch.no_grad()
    def sample(self, pixel_values, image_position_ids, prompt_ids, state, steps: int = 10):
        """Integrate the velocity field from noise back to an action chunk."""
        b = state.shape[0]
        x = torch.randn(b, HORIZON, ACTION_DIM, device=state.device, dtype=torch.bfloat16)
        prefix = self._prefix(pixel_values, image_position_ids, prompt_ids)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((b,), 1.0 - i * dt, device=state.device, dtype=x.dtype)
            hid = self._backbone(prefix, self._suffix(state, x, t))[:, -HORIZON:]
            v = self.action_out_proj(hid.to(torch.bfloat16))
            x = x - dt * v  # move from noise (t=1) toward the data (t=0)
        return x.float()

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]
