"""Image -> trajectory: generate FAST action tokens with the fine-tuned Gemma 4, decode them to
joint angles, and plot the prediction against the recorded ground truth."""
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

OUT = os.path.join(HERE, "out")
PROMPT = "<|image|>Predict the action trajectory.\n"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class OnlyActionTokens(LogitsProcessor):
    """Restrict generation to action tokens and eos, so every output is a valid FAST id."""

    def __init__(self, n_base: int, eos: int):
        self.n_base, self.eos = n_base, eos

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.n_base:] = scores[:, self.n_base:]
        mask[:, self.eos] = scores[:, self.eos]
        return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=os.path.join(OUT, "adapter_act"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--png", default=os.path.join(OUT, "trajectories_act.png"))
    ap.add_argument("--moving_only", action="store_true", help="only plot windows with real motion")
    args = ap.parse_args()

    base_ids = load_ids(OUT, "base_ids.json")
    proc = AutoProcessor.from_pretrained(MODEL)
    eos = base_ids.index(proc.tokenizer.eos_token_id)
    model, remap, n_base = load_compact_gemma4(base_ids)

    # The retrained embedding tables are not part of the LoRA delta, so restore them explicitly.
    emb = torch.load(os.path.join(args.adapter, "embeddings.pt"))
    tm = model.model.language_model
    tm.embed_tokens.weight.data = emb["embed_tokens"].to("cuda", torch.bfloat16)
    tm.embed_tokens_per_layer.weight.data = emb["embed_tokens_per_layer"].to("cuda", torch.bfloat16)
    model.tie_weights()
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    fast = AutoProcessor.from_pretrained("lerobot/fast-action-tokenizer", trust_remote_code=True)
    stats = np.load(os.path.join(OUT, "norm_stats.npz"))
    mean, std = stats["mean"], stats["std"]
    procs = LogitsProcessorList([OnlyActionTokens(n_base, eos)])

    pairs = json.load(open(os.path.join(OUT, "pairs.json")))
    if args.moving_only:  # a still arm compresses to very few tokens; those are the easy ones
        pairs = [p for p in pairs if len(p["tokens"]) >= 25]
    picks = pairs[:: max(1, len(pairs) // args.n)][: args.n]

    fig, axes = plt.subplots(len(picks), 2, figsize=(13, 3.1 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    maes = []

    for row, p in enumerate(picks):
        img = Image.open(os.path.join(OUT, p["img"])).convert("RGB")
        enc = proc(text=PROMPT, images=[img], return_tensors="pt")
        enc = {k: v.to("cuda") for k, v in enc.items()}
        enc["input_ids"] = remap[enc["input_ids"]]
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=70, do_sample=False,
                                 logits_processor=procs, eos_token_id=eos)
        new = gen[0, enc["input_ids"].shape[1]:].tolist()
        pred_ids = [t - n_base for t in new if t != eos]

        gt = fast.decode([p["tokens"]], time_horizon=50, action_dim=6)[0] * std + mean
        ok = False
        if pred_ids:
            # A malformed sequence does not raise: the FAST decoder catches it and returns zeros.
            raw = fast.decode([pred_ids], time_horizon=50, action_dim=6)[0]
            ok = bool(np.abs(raw).sum() > 0)
            pred = raw * std + mean
        if ok:
            maes.append(np.abs(pred - gt).mean())
        motion = gt.max(0) - gt.min(0)
        status = f"MAE {np.abs(pred - gt).mean():6.2f} deg" if ok else "DECODE FAIL"
        print(f"[{row}] ep={p['ep']:2d} | gen {len(pred_ids):2d} tok (target {len(p['tokens']):2d}) "
              f"| gt motion {motion.max():5.1f} deg | {status}")

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"input image (ep {p['ep']}, frame {p['start']})", fontsize=9)
        axes[row, 0].axis("off")
        ax = axes[row, 1]
        for j in range(6):
            (line,) = ax.plot(gt[:, j], lw=1.4, alpha=0.85, label=JOINTS[j] if row == 0 else None)
            if ok:
                ax.plot(pred[:, j], lw=1.4, ls="--", color=line.get_color())
        ax.set_title("solid = ground truth, dashed = Gemma 4 prediction", fontsize=9)
        ax.set_xlabel("step")
        ax.set_ylabel("deg")
        if row == 0:
            ax.legend(fontsize=6, ncol=3)

    if maes:
        print(f"\nmean MAE over {len(maes)}/{len(picks)} decoded: {np.mean(maes):.2f} deg")
    plt.tight_layout()
    plt.savefig(args.png, dpi=110)
    print("wrote", args.png)


if __name__ == "__main__":
    main()
