"""Build (image -> 50-step action trajectory as FAST tokens) pairs from the SO-101 dataset.

Outputs:
  out/images/{i:05d}.png            first-frame image of each window
  out/pairs.json                    list of {img, tokens, ep, start}
  out/norm_stats.npz                action mean/std for de/normalization
"""
import glob, json, os
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoProcessor
from lerobot.datasets import LeRobotDataset

ROOT = "/media/tsukasa/DATA/so101/pick_and_place"
REPO = "local/pick_and_place"
OUT = os.path.join(os.path.dirname(__file__), "out")
HORIZON = 50
STRIDE = 100  # windows per episode

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)

# --- actions from parquet (fast, no video decode) ---
files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("index").reset_index(drop=True)
actions = np.stack(df["action"].to_numpy())  # [N,6]
print("total frames:", len(actions), "action dim:", actions.shape[1])

# --- normalization stats + FAST tokenizer ---
ds = LeRobotDataset(REPO, root=ROOT)
st = ds.meta.stats["action"]
mean = np.array(st["mean"], dtype="float32")
std = np.array(st["std"], dtype="float32")
np.savez(os.path.join(OUT, "norm_stats.npz"), mean=mean, std=std)

tok = AutoProcessor.from_pretrained("lerobot/fast-action-tokenizer", trust_remote_code=True)

pairs = []
idx = 0
for ep in ds.meta.episodes:
    e_from, e_to = ep["dataset_from_index"], ep["dataset_to_index"]
    for start in range(e_from, e_to - HORIZON, STRIDE):
        chunk = actions[start:start + HORIZON][None]          # [1,50,6]
        norm = (chunk - mean) / (std + 1e-6)
        token_ids = tok(norm.astype("float32"))[0]            # list[int]
        # decode this frame's image via the dataset (video decode, 1 per window)
        img = ds[start]["observation.images.front"]           # CHW float [0,1]
        arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        Image.fromarray(arr).save(os.path.join(OUT, "images", f"{idx:05d}.png"))
        pairs.append({"img": f"images/{idx:05d}.png", "tokens": [int(t) for t in token_ids],
                      "ep": int(ep["episode_index"]), "start": int(start)})
        idx += 1

json.dump(pairs, open(os.path.join(OUT, "pairs.json"), "w"))
tok_lens = [len(p["tokens"]) for p in pairs]
print(f"wrote {len(pairs)} pairs | token len min/mean/max = "
      f"{min(tok_lens)}/{np.mean(tok_lens):.1f}/{max(tok_lens)} | max token id = "
      f"{max(max(p['tokens']) for p in pairs)}")
