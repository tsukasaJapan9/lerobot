"""Load Gemma 4 E2B in 4-bit on a 6GB GPU, with FAST action tokens added to its vocabulary.

Three tensors are sized by the vocabulary and together dwarf the rest of the model:

    embed_tokens_per_layer (PLE)  262144 x 8960  = 2.35B params = 4.7GB bf16
    embed_tokens / lm_head (tied) 262144 x 1536  = 0.40B params = 0.8GB bf16
    logits during the loss         seq x 262144  = ~0.5GB once cross-entropy upcasts to fp32

bitsandbytes does not quantize embeddings, so none of that shrinks with 4-bit, and accelerate
cannot offload it (a CPU-assigned module is still moved to the execution device for its forward).

The task only needs 12 real text tokens (the fixed prompt, the image placeholder, bos/eos/pad),
so we rebuild the model over a compact vocabulary of those 12 plus one dedicated token per FAST
action id. All three tensors above collapse to tens of MB, and an action chunk becomes one token
per FAST id instead of a long run of digit characters.

Compact id layout:  [0 .. n_base)  real text tokens (rows copied from the checkpoint)
                    [n_base .. )   action token i == FAST id (i - n_base), randomly initialized

Callers must map ids through `remap` before the model sees them, and subtract `n_base` from
generated ids to recover FAST ids.
"""
import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, BitsAndBytesConfig, Gemma4ForConditionalGeneration

MODEL = "google/gemma-4-E2B"
PLE_KEY = "model.language_model.embed_tokens_per_layer.weight"
EMB_KEY = "model.language_model.embed_tokens.weight"
# Vocab-sized tensors are rebuilt from scratch, so they must be *physically absent* from the
# checkpoint: leaving them in is not enough to skip them (transformers reads every tensor in a
# shard, and `ignore_mismatched_sizes` just rebuilds the module at the checkpoint's size).
DROP_KEYS = {PLE_KEY, EMB_KEY, "lm_head.weight"}
N_ACTION = 2048  # FAST tokenizer vocabulary


def _local_model_dir(base_ids: list[int], n_vocab: int) -> str:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(MODEL)
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "gemma4_act")
    os.makedirs(local, exist_ok=True)
    compact = {tid: i for i, tid in enumerate(base_ids)}

    for fn in os.listdir(snap):
        if fn in ("config.json", "model.safetensors", "generation_config.json"):
            continue
        dst = os.path.join(local, fn)
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(os.path.join(snap, fn)), dst)

    cfg = json.load(open(os.path.join(snap, "config.json")))
    cfg["text_config"]["vocab_size"] = n_vocab
    cfg["text_config"]["vocab_size_per_layer_input"] = n_vocab
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        cfg["text_config"][key] = compact.get(cfg["text_config"][key], 0)
    # Ids the model compares against input_ids; anything we never use goes out of range so the
    # comparison is simply always false.
    for key in ("image_token_id", "video_token_id", "audio_token_id", "boi_token_id",
                "eoi_token_id", "boa_token_id", "eoa_token_id"):
        if key in cfg:
            cfg[key] = compact.get(cfg[key], -1)
    json.dump(cfg, open(os.path.join(local, "config.json"), "w"))

    index_path = os.path.join(local, "model.safetensors.index.json")
    if os.path.exists(index_path):
        return local

    src = os.path.join(snap, "model.safetensors")
    weight_map, shard, nbytes, shard_i = {}, {}, 0, 1
    with safe_open(src, framework="pt", device="cpu") as f:

        def flush():
            nonlocal shard, nbytes, shard_i
            name = f"model-{shard_i:05d}.safetensors"
            save_file(shard, os.path.join(local, name), metadata={"format": "pt"})
            weight_map.update(dict.fromkeys(shard, name))
            shard, nbytes, shard_i = {}, 0, shard_i + 1

        for k in f.keys():
            if k in DROP_KEYS:
                continue
            t = f.get_tensor(k)
            shard[k] = t
            nbytes += t.numel() * t.element_size()
            if nbytes > 1_000_000_000:  # ~1GB shards keep host RAM low
                flush()
        if shard:
            flush()

    json.dump({"metadata": {"total_size": 0}, "weight_map": weight_map}, open(index_path, "w"))
    return local


def _init_vocab_rows(key: str, base_ids: list[int], n_vocab: int, wpath: str) -> torch.Tensor:
    """Real-token rows copied from the checkpoint; action-token rows drawn at the same scale."""
    with safe_open(wpath, framework="pt", device="cpu") as f:
        sl = f.get_slice(key)
        base = torch.stack([sl[i : i + 1][0] for i in base_ids]).float()
    rows = torch.empty(n_vocab, base.shape[1], dtype=torch.float32)
    rows[: len(base_ids)] = base
    torch.manual_seed(0)
    rows[len(base_ids):].normal_(mean=0.0, std=float(base.std()))
    return rows.to("cuda", torch.bfloat16)


def load_compact_gemma4(base_ids: list[int], n_action: int = N_ACTION):
    """Return (model, remap, n_base). Action token for FAST id t has compact id n_base + t."""
    n_base = len(base_ids)
    n_vocab = n_base + n_action
    local = _local_model_dir(base_ids, n_vocab)
    cfg = AutoConfig.from_pretrained(local)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Gemma4ForConditionalGeneration.from_pretrained(
        local, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}
    )

    from huggingface_hub import hf_hub_download

    wpath = hf_hub_download(MODEL, "model.safetensors")
    tm = model.model.language_model
    tm.embed_tokens.weight.data = _init_vocab_rows(EMB_KEY, base_ids, n_vocab, wpath)
    tm.embed_tokens_per_layer.weight.data = _init_vocab_rows(PLE_KEY, base_ids, n_vocab, wpath)
    model.tie_weights()  # lm_head shares embed_tokens

    remap = torch.zeros(262144, dtype=torch.long, device="cuda")
    remap[torch.tensor(base_ids, device="cuda")] = torch.arange(n_base, device="cuda")

    n_layers = cfg.text_config.num_hidden_layers
    hpl = cfg.text_config.hidden_size_per_layer_input

    def get_per_layer_inputs(input_ids, inputs_embeds=None):
        return tm.embed_tokens_per_layer(input_ids).reshape(*input_ids.shape, n_layers, hpl)

    tm.get_per_layer_inputs = get_per_layer_inputs

    gc = model.generation_config
    gc.bos_token_id = cfg.text_config.bos_token_id
    gc.eos_token_id = cfg.text_config.eos_token_id
    gc.pad_token_id = cfg.text_config.pad_token_id
    return model, remap, n_base


def load_ids(out_dir: str, name: str) -> list[int]:
    return json.load(open(os.path.join(out_dir, name)))
