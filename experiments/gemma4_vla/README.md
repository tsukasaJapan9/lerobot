# Gemma 4 as a VLA: image → robot trajectory on a 6GB laptop GPU

Fine-tuning **Gemma 4 E2B** (Google's multimodal model, released 2026-04) into a
vision-language-action model: given one camera frame plus the arm's joint state, predict the next
50 steps of SO-101 joint angles. Offline only — no real-time control, so slow inference is fine.

Ran on an **RTX 4050 Laptop (6GB)** against 50 teleoperated episodes of a pick-and-place task
(44,804 frames) recorded with `lerobot-record`.

**Where it landed:** the model produces smooth trajectories that move in the right direction
(shape correlation +0.33, held-out MAE 14.8°), but consistently undershoots the magnitude. Not
deployable, but qualitatively doing the right thing. Getting there took six attempts, and the four
dead ends are the more instructive half — each one is documented below, because they are the sort
of thing that looks like it should work and quietly does not.

## Part 1 — Fitting Gemma 4 E2B into 6GB

Gemma 4 E2B advertises "2.3B effective params, <1.5GB footprint, runs on a Raspberry Pi 5". That
footprint is for an optimized inference runtime. In plain `transformers` the checkpoint is
**10.25GB of bf16 weights**, and the problem is not the transformer stack — it is the three tensors
sized by the 262144-token vocabulary:

| tensor | shape | size (bf16) |
| --- | --- | --- |
| `embed_tokens_per_layer` (Per-Layer Embeddings) | 262144 × 8960 | **4.7GB** |
| `embed_tokens` / `lm_head` (tied) | 262144 × 1536 | 0.8GB |
| logits during the loss | seq × 262144 | ~0.5GB (fp32 after the cross-entropy upcast) |

Neither 4-bit quantization nor offloading helps:

- **bitsandbytes does not quantize embeddings.** Only `nn.Linear` is converted, so the PLE table
  stays bf16 and alone overflows the GPU.
- **accelerate cannot offload it.** A module assigned to `"cpu"` in a `device_map` is still moved
  to the execution device for its forward pass (`AlignDevicesHook.init_hook` →
  `set_module_tensor_to_device(..., execution_device)`), so a 4.7GB table gets copied onto a 6GB
  GPU and OOMs. `device_map="auto"` with a `max_memory` cap just pushed *everything* to the CPU.

**What works: compact the vocabulary.** The prompt is fixed and the actions are continuous, so the
model only ever embeds ~12 real tokens. Rebuild it with `vocab_size = 12`, copy those rows out of
the original checkpoint, and remap ids on the way in. All three tensors collapse to a few hundred KB.

```
peak GPU: OOM → 2.3GB     (4-bit base + LoRA + gradient checkpointing, batch 1)
```

Two undocumented `transformers` behaviours had to be worked around (`model_utils.py`):

1. **`ignore_mismatched_sizes=True` rebuilds the module at the *checkpoint's* size.** It does not
   skip the oversized tensor — it materializes it (on the GPU) and then resizes the module to
   match, putting the 4.7GB table right back.
2. **Omitting a key from `model.safetensors.index.json` does not skip it.** transformers reads
   every tensor present in a shard file, and a file named exactly `model.safetensors` is loaded
   directly, ignoring the index. The vocab-sized tensors must be *physically absent*, so
   `_local_model_dir()` republishes the weights as ~1GB shards with those keys dropped.

## Part 2 — Six attempts at predicting a trajectory

| v | approach | outcome |
| --- | --- | --- |
| v1 | FAST action tokens written as **digit text** (`"127 222 1024"`) | Only static windows worked; moving ones degenerated into repetition (`392 392 392…`) and failed to decode. |
| v2 | FAST tokens as **dedicated vocab tokens** | **FAST is BPE over DCT coefficients — variable-length.** A valid token sequence still has to expand to exactly 50×6=300 coefficients, and usually did not. Half the samples failed to decode. |
| v3 | **fixed-length bins** (10 waypoints × 6 joints = always 60 tokens) | Decode failures structurally eliminated. But every prediction was a **flat line**, and held-out MAE *worsened* (26°→32°) while the training loss kept falling. |
| v4 | + the arm's **current joint state** in the prompt | The trajectory now starts at the right pose — the state input is demonstrably used — but the model then holds it constant. It learned "repeat the current state". |
| v5 | **pi0's design**: continuous actions, flow matching, trainable vision | Flat lines broken, but replaced by **flat + high-frequency noise**, shape correlation ≈ 0. Motion, but random motion. |
| **v6** | + pi0's **block attention mask** | Smooth curves, every correlation positive (**+0.33 mean**, best +0.60), MAE **14.8°**. Direction right, magnitude undershoots. |

### Why v1–v4 always went flat

All four generated actions as **discrete tokens, autoregressively, under greedy decoding**. Greedy
decoding takes the safest next token at every step, and when the future is ambiguous — a single
frame cannot distinguish "descending" from "ascending" — the safest answer is *don't move*. The
model was minimizing its loss correctly; the failure was in the objective, not the optimization.

Two further handicaps, both fixed in v5:

- **No proprioceptive state (v1–v3).** ACT, pi0 and SmolVLA all take joint angles alongside the
  image. Without them the model had to read the arm's pose off a top-down photo.
- **A frozen vision tower.** PEFT cannot adapt the vision tower's `Gemma4ClippableLinear`, so v1–v4
  left it frozen and the visual features never learned what decides where the arm goes. But **the
  real `nn.Linear` sits one level in, as `.linear`**, and PEFT does support `Linear4bit` — so LoRA
  can reach the vision encoder after all (112 adapters). pi0 ships `freeze_vision_encoder=False`
  for exactly this reason.

  Watch out for a silent failure here: with `use_reentrant=True` gradient checkpointing, the vision
  LoRA receives **no gradients at all**, because the reentrant variant needs an input that requires
  grad and `pixel_values` does not. `gemma4_pi0.py` passes `use_reentrant=False` and the test
  asserts that the adapters actually get non-zero grads. (Check `lora_B`, not `lora_A` — `lora_B`
  is zero-initialised, so `lora_A`'s gradient is necessarily zero at step 0 and looks like a bug.)

### Why v5 was noisy and v6 is not

v5 fed the noisy action chunk through the LM under its **default causal mask**, so action step *i*
could only see steps `0..i`. A denoiser has to see the *whole* noisy chunk to correct it coherently;
under causal masking each step is denoised almost independently, which is exactly what
high-frequency jitter looks like.

pi0 instead uses a block mask (`make_att_2d_masks`), reproduced in `Gemma4FlowPolicy._block_mask`:

```
       pre0  pre1  pre2 state  act0  act1  act2  act3
 pre0     o     o     o     .     .     .     .     .    prefix: bidirectional, cannot see actions
state     o     o     o     o     .     .     .     .    state: prefix + itself
 act0     o     o     o     o     o     o     o     o    actions: everything, and each other
 act3     o     o     o     o     o     o     o     o
```

`Gemma4TextModel.forward` builds its own causal mask unless `attention_mask` is passed **as a
dict** (`{"full_attention": mask, "sliding_attention": mask}`) — that is the escape hatch.

Telling detail: v6's *training loss is higher* than v5's while its trajectories are far better.
Causal masking let v5 denoise each step from its own local noise and score well on MSE without ever
learning the global shape. **The loss was not the thing to optimize for** — which is why every
version here is scored on shape correlation, on a held-out split.

### What is still wrong

v6 gets the direction right and consistently **undershoots the magnitude** — where the recording
sweeps to −105°, it stops around −60°. That is a denoiser hedging toward the mean, the signature of
too little data (1,400 windows; pi0 and SmolVLA are pretrained on hundreds to thousands of hours of
robot data). It is now a question of scale, not of design.

## Files

| file | role |
| --- | --- |
| `model_utils.py` | **the reusable piece** — loads Gemma 4 E2B in 4-bit on 6GB via vocabulary compaction |
| `gemma4_pi0.py` | **the working policy** — pi0-style flow matching on a Gemma 4 backbone, with the block attention mask and LoRA reaching into the vision tower |
| `prep_v5.py`, `train_v5.py`, `infer_v5.py` | v5/v6: continuous action chunks, training, held-out evaluation with the shape-correlation metric |
| `prep_data_v3.py`, `add_state.py` | v3/v4 data: fixed-length binned trajectories, then joint state |
| `train_v3.py`, `train_v4.py`, `infer_v3.py`, `infer_v4.py` | v3/v4, kept as documented negative results |
| `prep_data.py`, `train_qlora.py`, `infer_plot.py` | v1/v2, the FAST/BPE approach — likewise |

## Running it

Needs `lerobot[feetech,viz,dataset,training,pi,peft]` plus `bitsandbytes` and `matplotlib`. The
dataset path is hardcoded at the top of `prep_data_v3.py` / `prep_v5.py` — point `ROOT` at your own
`LeRobotDataset`.

```bash
uv run python experiments/gemma4_vla/prep_data_v3.py    # decode one frame per window
uv run python experiments/gemma4_vla/prep_v5.py         # continuous action chunks + state
uv run python experiments/gemma4_vla/train_v5.py --n_samples 1400 --epochs 6
uv run python experiments/gemma4_vla/infer_v5.py --ckpt out_v5/ckpt_ep5 --n 6
```

About 4 hours for 6 epochs over 1,400 windows on an RTX 4050 (batch 1, grad-accum 8). Checkpoints
are written and scored on the held-out split every epoch, because v3 showed the training loss
falling happily while generalisation got worse — the last checkpoint is not necessarily the one to
keep.
