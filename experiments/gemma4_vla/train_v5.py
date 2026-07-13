"""Train the pi0-style flow-matching policy on a Gemma 4 E2B backbone.

Batch size is 1: the vision tower returns its tokens packed as [n_tokens, H] with a per-image
count, so batching images would need unpacking. Gradient accumulation stands in for it.

Held-out MAE is reported every epoch, because v3 showed the training loss falling happily while
generalisation got worse -- the last checkpoint is not necessarily the one to keep.
"""
import argparse, json, os, random, sys, time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma4_pi0 import Gemma4FlowPolicy, LORA_TARGETS  # noqa: E402
from model_utils import MODEL, load_ids  # noqa: E402

OUT = os.path.join(HERE, "out_v5")
PROMPT = "<|image|>Predict the action trajectory.\n"


class Batch:
    """Pre-processes one sample: image -> pixel values, plus the continuous state/action arrays."""

    def __init__(self, proc, remap, images, npz):
        self.proc, self.remap, self.images = proc, remap, images
        self.actions = torch.tensor(npz["actions"])
        self.state = torch.tensor(npz["state"])

    def __len__(self):
        return len(self.images)

    def get(self, i):
        img = Image.open(self.images[i]).convert("RGB")
        enc = self.proc(text=PROMPT, images=[img], return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        return dict(
            pixel_values=enc["pixel_values"],
            image_position_ids=enc["image_position_ids"],
            prompt_ids=self.remap[enc["input_ids"][:, -8:]],
            state=self.state[i : i + 1].to("cuda", torch.bfloat16),
            actions=self.actions[i : i + 1].to("cuda", torch.bfloat16),
        )


@torch.no_grad()
def evaluate(pol, val, norm, n=20, steps=10):
    """Denoise a trajectory per held-out sample and report MAE in degrees."""
    pol.eval()
    a_mean, a_std = norm["a_mean"], norm["a_std"]
    errs = []
    for i in range(min(n, len(val))):
        b = val.get(i)
        pred = pol.sample(b["pixel_values"], b["image_position_ids"], b["prompt_ids"],
                          b["state"], steps=steps)[0].cpu().numpy()
        gt = b["actions"][0].float().cpu().numpy()
        errs.append(np.abs(pred * a_std - gt * a_std).mean())  # de-normalised, in degrees
    pol.train()
    return float(np.mean(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=1400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--out", type=str, default=os.path.join(OUT, "ckpt"))
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL)
    pol = Gemma4FlowPolicy(load_ids(os.path.join(HERE, "out_v3"), "base_ids.json"))
    pol.vlm = get_peft_model(pol.vlm, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", target_modules=LORA_TARGETS))
    pol.train()

    tr = json.load(open(os.path.join(OUT, "pairs.json")))
    va = json.load(open(os.path.join(OUT, "val.json")))
    train = Batch(proc, pol.remap, tr["images"], np.load(os.path.join(OUT, "pairs.npz")))
    val = Batch(proc, pol.remap, va["images"], np.load(os.path.join(OUT, "val.npz")))
    norm = np.load(os.path.join(OUT, "norm.npz"))

    params = [p for p in pol.parameters() if p.requires_grad]
    print(f"train {len(train)} | val {len(val)} | trainable {sum(p.numel() for p in params)/1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=args.lr)

    idx = list(range(min(args.n_samples, len(train))))
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        random.shuffle(idx)
        run, n = 0.0, 0
        for k, i in enumerate(idx):
            b = train.get(i)
            loss = pol.loss(b["pixel_values"], b["image_position_ids"], b["prompt_ids"],
                            b["state"], b["actions"])
            (loss / args.grad_accum).backward()
            run += float(loss)
            n += 1
            if (k + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"ep {ep} step {step} loss {run / n:.4f} "
                          f"peak_mem {torch.cuda.max_memory_allocated()/1e9:.2f}GB "
                          f"elapsed {time.time()-t0:.0f}s", flush=True)
                    run, n = 0.0, 0

        mae = evaluate(pol, val, norm)
        ckpt = f"{args.out}_ep{ep}"
        os.makedirs(ckpt, exist_ok=True)
        pol.vlm.save_pretrained(ckpt)
        torch.save({k: v for k, v in pol.state_dict().items()
                    if k.startswith(("state_proj", "action_"))}, os.path.join(ckpt, "head.pt"))
        print(f"[ep {ep}] held-out MAE {mae:.2f} deg -> {ckpt}", flush=True)

    print("done")


if __name__ == "__main__":
    main()
