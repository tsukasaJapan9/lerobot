"""QLoRA fine-tune Gemma 4 E2B: (image + current joint state) -> fixed-length binned trajectory.

The answer is always WAYPOINTS*6 = 60 tokens: position w*6+j is joint j at waypoint w, and each
token is one of 256 bins, so the output can never be malformed.

Unlike v3, the prompt also carries the arm's current joint angles as 6 bin tokens. Without them the
model had to read joint angles off a top-down photo and could not tell "descending" from
"ascending", so the loss-minimising answer was the average future -- a flat trajectory, which is
exactly what it learned to emit.
"""
import argparse, json, os, random, sys, time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_utils import MODEL, load_compact_gemma4, load_ids  # noqa: E402

OUT = os.path.join(HERE, "out_v3")
PROMPT = "<|image|>Predict the action trajectory.\n"
N_BINS = 256


def build_example(proc, img_path, bins, state_bins, remap, n_base):
    img = Image.open(img_path).convert("RGB")
    enc = proc(text=PROMPT, images=[img], return_tensors="pt")
    enc = {k: v.to("cuda") for k, v in enc.items()}
    prefix = remap[enc["input_ids"]]

    state = torch.tensor([[n_base + b for b in state_bins]], device="cuda")  # [1, 6]
    answer = torch.tensor([[n_base + b for b in bins]], device="cuda")  # [1, 60]
    tail = torch.cat([state, answer], dim=1)
    enc["input_ids"] = torch.cat([prefix, tail], dim=1)
    enc["attention_mask"] = torch.cat([enc["attention_mask"], torch.ones_like(tail)], dim=1)
    enc["mm_token_type_ids"] = torch.cat([enc["mm_token_type_ids"], torch.zeros_like(tail)], dim=1)
    labels = enc["input_ids"].clone()
    labels[:, : prefix.shape[1] + state.shape[1]] = -100  # supervise the trajectory only
    enc["labels"] = labels
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=1400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--out", type=str, default=os.path.join(OUT, "adapter_v4"))
    args = ap.parse_args()

    pairs = json.load(open(os.path.join(OUT, "pairs.json")))
    random.seed(0)
    random.shuffle(pairs)
    val, pairs = pairs[:40], pairs[40 : 40 + args.n_samples]  # val.json is fixed; see add_state.py
    print(f"train {len(pairs)} | held-out val {len(val)}")

    base_ids = load_ids(OUT, "base_ids.json")
    proc = AutoProcessor.from_pretrained(MODEL)
    model, remap, n_base = load_compact_gemma4(base_ids, n_action=N_BINS)

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Language-model layers only: the vision tower wraps its Linear4bit in a custom
        # Gemma4ClippableLinear that PEFT cannot adapt.
        target_modules=r".*language_model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
    model = get_peft_model(model, lora)

    # Bin tokens are new, so their embedding rows carry all their meaning and must be learned.
    tm = model.base_model.model.model.language_model
    embeds = [tm.embed_tokens.weight, tm.embed_tokens_per_layer.weight]
    for w in embeds:
        w.requires_grad_(True)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        random.shuffle(pairs)
        run_loss, n = 0.0, 0
        for i, p in enumerate(pairs):
            enc = build_example(proc, os.path.join(OUT, p["img"]), p["bins"], p["state_bins"], remap, n_base)
            out = model(**enc)
            (out.loss / args.grad_accum).backward()
            run_loss += out.loss.item()
            n += 1
            if (i + 1) % args.grad_accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0:
                    print(f"ep {ep} step {step} loss {run_loss / n:.4f} "
                          f"peak_mem {torch.cuda.max_memory_allocated() / 1e9:.2f}GB "
                          f"elapsed {time.time() - t0:.0f}s", flush=True)
                run_loss, n = 0.0, 0

        # One directory per epoch: v3 showed that training loss keeps falling while held-out
        # accuracy degrades, so the last checkpoint is not necessarily the one to keep.
        ckpt = f"{args.out}_ep{ep}"
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        torch.save({"embed_tokens": embeds[0].detach().cpu(),
                    "embed_tokens_per_layer": embeds[1].detach().cpu()},
                   os.path.join(ckpt, "embeddings.pt"))
        print(f"[ep {ep}] checkpoint saved -> {ckpt}", flush=True)

    json.dump({"prompt": PROMPT, "n_base": n_base, "n_bins": N_BINS},
              open(os.path.join(args.out, "fmt.json"), "w"))
    print("done ->", args.out)


if __name__ == "__main__":
    main()
