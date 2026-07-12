# Gemma 4 as a VLA: image → robot trajectory on a 6GB laptop GPU

An attempt to fine-tune **Gemma 4 E2B** (Google's multimodal 2.3B-effective model, released
2026-04) into a vision-language-action model: give it one camera frame, get back a 50-step SO-101
joint trajectory. Offline only — no real-time control, so slow inference is acceptable.

Ran on an **RTX 4050 Laptop (6GB)** against 50 teleoperated episodes of a pick-and-place task
(44,804 frames) recorded with `lerobot-record`.

**Outcome, up front: the systems half worked, the learning half did not.** Gemma 4 does fit and
train on 6GB, the pipeline runs end to end, but the model never learned to predict *motion* — it
emits flat trajectories. The failure was narrowed down over four iterations and the likely root
cause is identified below. Both halves are written up honestly because the negative results are
the more useful part.

## Part 1 — Fitting Gemma 4 E2B into 6GB (this worked)

Gemma 4 E2B is advertised at "2.3B effective params, <1.5GB footprint, runs on a Raspberry Pi 5".
That footprint is for an optimized/quantized inference runtime. In plain `transformers` the
checkpoint is **10.25GB of bf16 weights**, and the problem is not the transformer stack — it is the
three tensors sized by the 262144-token vocabulary:

| tensor | shape | size (bf16) |
| --- | --- | --- |
| `embed_tokens_per_layer` (Per-Layer Embeddings) | 262144 × 8960 | **4.7GB** |
| `embed_tokens` / `lm_head` (tied) | 262144 × 1536 | 0.8GB |
| logits during the loss | seq × 262144 | ~0.5GB (fp32 after cross-entropy upcast) |

Neither 4-bit quantization nor offloading helps:

- **bitsandbytes does not quantize embeddings.** Only `nn.Linear` is converted, so the PLE table
  stays bf16 and alone overflows the GPU.
- **accelerate cannot offload it.** A module assigned to `"cpu"` in a `device_map` still gets
  moved to the execution device for its forward pass (`AlignDevicesHook.init_hook` →
  `set_module_tensor_to_device(..., execution_device)`), so a 4.7GB table is copied to a 6GB GPU
  and OOMs. `device_map="auto"` with a `max_memory` cap just pushed *everything* to CPU instead.

### What actually works: compact the vocabulary

The prompt is fixed and the answer alphabet is small, so the task only ever touches **12 real text
tokens** (prompt words, image placeholder, bos/eos/pad). Rebuild the model over a compact
vocabulary of those 12 plus one token per action bin, copy only the corresponding rows out of the
original checkpoint, and remap ids on the way in. All three tensors collapse to tens of MB.

```
peak GPU: OOM  →  2.4GB     (4-bit base + LoRA + gradient checkpointing, batch 1)
```

Two undocumented `transformers` behaviours had to be worked around to get there
(see `model_utils.py`):

1. **`ignore_mismatched_sizes=True` rebuilds the module at the *checkpoint's* size.** It does not
   skip the oversized tensor — it materializes it (on the GPU) and then resizes the module to
   match, putting the 4.7GB table right back.
2. **Omitting a key from `model.safetensors.index.json` does not skip it.** transformers reads
   every tensor present in a shard file. And a file named exactly `model.safetensors` is loaded
   directly, ignoring the index entirely.

So the vocab-sized tensors must be **physically absent** from the checkpoint. `_local_model_dir()`
republishes the weights as ~1GB shards with those keys dropped (streamed, to keep host RAM low),
alongside a `config.json` declaring the smaller vocab. The rows we need are then read back from the
original checkpoint and assigned directly.

`get_per_layer_inputs()` is patched to look up the compact PLE table — it is the single place the
PLE is read, and Gemma 4's forward conveniently accepts a precomputed `per_layer_inputs`.

## Part 2 — Teaching it to predict trajectories (this did not work)

Four iterations, each narrowing the failure:

| v | action representation | input | result |
| --- | --- | --- | --- |
| v1 | FAST tokens written as **digit text** (`"127 222 1024"`) | image | Only static windows worked. Moving ones degenerated into repetition (`392 392 392…`) and failed to decode. |
| v2 | FAST tokens as **dedicated vocab tokens** (2048) | image | Still failed. **FAST is BPE over DCT coefficients — it is variable-length**, so a valid token sequence still has to expand to exactly 50×6=300 coefficients, and usually did not. Half the samples failed to decode; the rest were worse than predicting the mean. |
| v3 | **fixed-length bins**: 10 waypoints × 6 joints = always 60 tokens, 256 bins each | image | Decode failures **structurally eliminated** (0/6). But predictions were flat lines, and held-out MAE *worsened* (26°→32°) while training loss kept falling — plain overfitting. |
| v4 | same | image **+ current joint state** (6 bin tokens) | The trajectory now **starts at the right pose** — the state input is demonstrably used. But the model then holds it constant: it learned the degenerate solution "repeat the current state". |

Held-out MAE never got below ~26°, and no version ever produced trajectory *shape*.

### Why it stayed flat

ACT, trained on this same dataset, reaches ~50% real-robot success — so the task **is** learnable
from a single frame plus joint state. The decisive difference:

|  | ACT | this experiment |
| --- | --- | --- |
| vision encoder | ResNet18, **trained on the task** | Gemma 4's tower, **frozen and 4-bit** |

LoRA was applied to the language layers only — PEFT cannot wrap the vision tower's custom
`Gemma4ClippableLinear`, so it was excluded. The visual features therefore never adapt to the
robot, the model gets no usable signal about *where the arm should go*, and it falls back on
whatever minimizes loss without vision: a constant trajectory (v3: the mean/end pose; v4: a copy of
the given state).

This matches `AGENT_GUIDE.md`, which notes for SmolVLA that **unfreezing the vision encoder
"usually improves performance substantially"**. The untried v5 would keep the vision tower in bf16
and train it (≈+1.8GB, likely still within 6GB).

The honest caveat: with 1,360 training samples and a frozen backbone, adapting a 2.3B model into a
VLA is a steep ask. π0/SmolVLA work because they are *pretrained on hundreds to thousands of hours*
of robot data. Fine-tuning ACT on more episodes remains the shortest path to a working policy on
this hardware.

## Files

| file | role |
| --- | --- |
| `model_utils.py` | **the reusable piece** — loads Gemma 4 E2B in 4-bit on 6GB via vocabulary compaction |
| `prep_data_v3.py` | builds (image → fixed-length binned trajectory) pairs, rebalanced toward windows with real motion |
| `add_state.py` | adds the current joint state to each pair (v4) |
| `train_v4.py` / `infer_v4.py` | final version: (image + state) → trajectory, with held-out evaluation |
| `train_v3.py` / `infer_v3.py` | ablation: image only, no joint state |
| `prep_data.py` / `train_qlora.py` / `infer_plot.py` | v1/v2, the FAST/BPE approach — kept as a documented negative result |

## Running it

Needs `lerobot[feetech,viz,dataset,training,pi,peft]` plus `bitsandbytes` and `matplotlib`.
The dataset path is hardcoded at the top of `prep_data_v3.py` — point `ROOT` at your own
`LeRobotDataset`.

```bash
uv run python experiments/gemma4_vla/prep_data_v3.py      # build pairs (decodes one frame per window)
uv run python experiments/gemma4_vla/add_state.py         # attach joint state
uv run python experiments/gemma4_vla/train_v4.py --n_samples 1360 --epochs 6
uv run python experiments/gemma4_vla/infer_v4.py --adapter out_v3/adapter_v4_ep2 --n 6
```

Training is ~1.2 s/sample on an RTX 4050; checkpoints are written per epoch because, as v3 showed,
**the last one is not necessarily the best** — always score them on the held-out split.
