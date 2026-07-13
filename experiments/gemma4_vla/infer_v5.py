"""Denoise a trajectory from a held-out image and plot it against the recording.

The number to distrust is MAE: v1-v4 scored 26 deg while emitting perfectly flat lines, because a
constant near the middle of the motion is not far off in absolute terms. What matters is whether
the dashed curve *bends the way the solid one does*.
"""
import argparse, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma4_pi0 import Gemma4FlowPolicy  # noqa: E402
from model_utils import MODEL, load_ids  # noqa: E402

OUT = os.path.join(HERE, "out_v5")
PROMPT = "<|image|>Predict the action trajectory.\n"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--png", default=os.path.join(OUT, "traj_v5.png"))
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(MODEL)
    pol = Gemma4FlowPolicy(load_ids(os.path.join(HERE, "out_v3"), "base_ids.json"))
    pol.vlm = PeftModel.from_pretrained(pol.vlm, args.ckpt)
    pol.load_state_dict(torch.load(os.path.join(args.ckpt, "head.pt")), strict=False)
    pol.eval()

    norm = np.load(os.path.join(OUT, "norm.npz"))
    a_mean, a_std = norm["a_mean"], norm["a_std"]
    va = json.load(open(os.path.join(OUT, "val.json")))
    z = np.load(os.path.join(OUT, "val.npz"))

    order = sorted(range(len(va["images"])), key=lambda i: -va["meta"][i]["motion"])
    picks = order[: args.n]

    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 3.1 * len(picks)))
    maes, shape_scores = [], []
    for row, i in enumerate(picks):
        img = Image.open(va["images"][i]).convert("RGB")
        enc = proc(text=PROMPT, images=[img], return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        pid = pol.remap[enc["input_ids"][:, -8:]]
        st = torch.tensor(z["state"][i : i + 1]).to("cuda", torch.bfloat16)
        with torch.no_grad():
            pred = pol.sample(enc["pixel_values"], enc["image_position_ids"], pid,
                              st, steps=args.steps)[0].cpu().numpy()
        pred = pred * a_std + a_mean
        gt = z["actions"][i] * a_std + a_mean

        mae = float(np.abs(pred - gt).mean())
        maes.append(mae)
        # Does the prediction actually move, and does it move *with* the recording? A flat line
        # scores 0 range and undefined correlation; that is the failure v1-v4 kept hitting.
        rng = float((pred.max(0) - pred.min(0)).max())
        gt_rng = float((gt.max(0) - gt.min(0)).max())
        corr = float(np.mean([np.corrcoef(pred[:, j], gt[:, j])[0, 1]
                              for j in range(6) if pred[:, j].std() > 1e-3 and gt[:, j].std() > 1e-3]
                             or [0.0]))
        shape_scores.append(corr)
        print(f"[{row}] motion gt {gt_rng:6.1f} deg | pred range {rng:6.1f} deg "
              f"| corr {corr:+.2f} | MAE {mae:6.2f} deg")

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"held-out image (ep {va['meta'][i]['ep']})", fontsize=9)
        axes[row, 0].axis("off")
        ax = axes[row, 1]
        for j in range(6):
            (line,) = ax.plot(gt[:, j], lw=1.4, label=JOINTS[j] if row == 0 else None)
            ax.plot(pred[:, j], lw=1.4, ls="--", color=line.get_color())
        ax.set_title(f"solid = ground truth, dashed = Gemma 4  "
                     f"(MAE {mae:.1f} deg, shape corr {corr:+.2f})", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("deg")
        if row == 0:
            ax.legend(fontsize=6, ncol=3)

    print(f"\nmean MAE {np.mean(maes):.2f} deg | mean shape correlation {np.mean(shape_scores):+.2f}")
    plt.tight_layout()
    plt.savefig(args.png, dpi=110)
    print("wrote", args.png)


if __name__ == "__main__":
    main()
