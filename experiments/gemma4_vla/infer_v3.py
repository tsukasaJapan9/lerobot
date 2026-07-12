"""Image -> trajectory with the v3 (fixed-length binned) model, evaluated on held-out windows."""
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, LogitsProcessorList

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_utils import MODEL, load_compact_gemma4, load_ids  # noqa: E402

OUT = os.path.join(HERE, "out_v3")
PROMPT = "<|image|>Predict the action trajectory.\n"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
N_BINS = 256


class OnlyBins(LogitsProcessor):
    """Only bin tokens may be generated, so the output always decodes."""

    def __init__(self, n_base):
        self.n_base = n_base

    def __call__(self, input_ids, scores):
        out = torch.full_like(scores, float("-inf"))
        out[:, self.n_base:] = scores[:, self.n_base:]
        return out


def unbin(bins, lo, hi):
    span = np.maximum(hi - lo, 1e-6)
    return np.asarray(bins).reshape(-1, 6) / (N_BINS - 1) * span + lo  # [W, 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(OUT, "adapter"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--png", default=os.path.join(OUT, "traj_v3.png"))
    args = ap.parse_args()

    s = np.load(os.path.join(OUT, "bin_stats.npz"))
    lo, hi, idx = s["lo"], s["hi"], s["idx"]

    base_ids = load_ids(OUT, "base_ids.json")
    proc = AutoProcessor.from_pretrained(MODEL)
    model, remap, n_base = load_compact_gemma4(base_ids, n_action=N_BINS)

    emb = torch.load(os.path.join(args.adapter, "embeddings.pt"))
    tm = model.model.language_model
    tm.embed_tokens.weight.data = emb["embed_tokens"].to("cuda", torch.bfloat16)
    tm.embed_tokens_per_layer.weight.data = emb["embed_tokens_per_layer"].to("cuda", torch.bfloat16)
    model.tie_weights()
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    procs = LogitsProcessorList([OnlyBins(n_base)])

    val = json.load(open(os.path.join(OUT, "val.json")))  # never seen during training
    val.sort(key=lambda p: -p["motion"])  # show the ones with real motion first
    picks = val[: args.n]

    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 3.1 * len(picks)))
    maes = []
    for row, p in enumerate(picks):
        img = Image.open(os.path.join(OUT, p["img"])).convert("RGB")
        enc = proc(text=PROMPT, images=[img], return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        enc["input_ids"] = remap[enc["input_ids"]]
        n_tok = len(p["bins"])
        with torch.no_grad():
            gen = model.generate(**enc, min_new_tokens=n_tok, max_new_tokens=n_tok,
                                 do_sample=False, logits_processor=procs)
        pred_bins = [t - n_base for t in gen[0, enc["input_ids"].shape[1]:].tolist()]

        pred = unbin(pred_bins, lo, hi)
        gt = unbin(p["bins"], lo, hi)
        mae = float(np.abs(pred - gt).mean())
        maes.append(mae)
        print(f"[{row}] ep={p['ep']:2d} motion={p['motion']:6.1f} deg | MAE {mae:6.2f} deg")

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"held-out image (ep {p['ep']}, frame {p['start']})", fontsize=9)
        axes[row, 0].axis("off")
        ax = axes[row, 1]
        for j in range(6):
            (line,) = ax.plot(idx, gt[:, j], lw=1.5, marker="o", ms=3,
                              label=JOINTS[j] if row == 0 else None)
            ax.plot(idx, pred[:, j], lw=1.5, ls="--", marker="x", ms=4, color=line.get_color())
        ax.set_title(f"solid = ground truth, dashed = Gemma 4  (MAE {mae:.1f} deg)", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("deg")
        if row == 0:
            ax.legend(fontsize=6, ncol=3)

    print(f"\nmean MAE on held-out: {np.mean(maes):.2f} deg")
    plt.tight_layout()
    plt.savefig(args.png, dpi=110)
    print("wrote", args.png)


if __name__ == "__main__":
    main()
