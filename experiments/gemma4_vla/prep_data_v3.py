"""Build (image -> fixed-length binned trajectory) pairs.

FAST/BPE was the wrong representation for autoregressive generation: it is variable-length, so
the model had to land on a token sequence that expands to exactly 50x6 DCT coefficients, and it
usually did not (half the samples failed to decode at all).

Here the answer is always exactly WAYPOINTS*6 tokens: waypoint w, joint j sits at position w*6+j,
and each token is a 256-way bin of that joint's angle. Decoding cannot fail.

The window set is also rebalanced: most of the recording is the arm sitting still, and training on
that as-is collapsed the model onto "predict a flat trajectory".
"""
import glob, json, os

import numpy as np
import pandas as pd
from PIL import Image
from lerobot.datasets import LeRobotDataset

ROOT = "/media/tsukasa/DATA/so101/pick_and_place"
REPO = "local/pick_and_place"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_v3")
HORIZON = 50
WAYPOINTS = 10
STRIDE = 10
N_BINS = 256
MOVING_DEG = 15.0  # a window counts as "moving" if some joint sweeps at least this much
STATIC_FRAC = 0.2  # keep a minority of still windows so the model still learns to hold position
MAX_SAMPLES = 1400

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
rng = np.random.default_rng(0)

files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("index").reset_index(drop=True)
actions = np.stack(df["action"].to_numpy()).astype("float32")  # [N,6]

ds = LeRobotDataset(REPO, root=ROOT)
st = ds.meta.stats["action"]
lo = np.array(st["min"], dtype="float32")
hi = np.array(st["max"], dtype="float32")
span = np.maximum(hi - lo, 1e-6)
idx = np.linspace(0, HORIZON - 1, WAYPOINTS).round().astype(int)

# 1. enumerate windows and score their motion (no video decode yet)
cand = []
for ep in ds.meta.episodes:
    for start in range(ep["dataset_from_index"], ep["dataset_to_index"] - HORIZON, STRIDE):
        win = actions[start : start + HORIZON]
        motion = float((win.max(0) - win.min(0)).max())
        cand.append((start, int(ep["episode_index"]), motion))

moving = [c for c in cand if c[2] >= MOVING_DEG]
static = [c for c in cand if c[2] < MOVING_DEG]
print(f"windows: {len(cand)} total | {len(moving)} moving | {len(static)} static")

n_static = min(len(static), int(MAX_SAMPLES * STATIC_FRAC))
n_moving = min(len(moving), MAX_SAMPLES - n_static)
sel = [moving[i] for i in rng.choice(len(moving), n_moving, replace=False)]
sel += [static[i] for i in rng.choice(len(static), n_static, replace=False)]
rng.shuffle(sel)
print(f"selected {len(sel)}: {n_moving} moving + {n_static} static")

# 2. decode one image per selected window and bin its trajectory
pairs = []
for i, (start, ep_i, motion) in enumerate(sel):
    way = actions[start + idx]  # [WAYPOINTS, 6]
    bins = np.clip(((way - lo) / span * (N_BINS - 1)).round(), 0, N_BINS - 1).astype(int)
    img = ds[start]["observation.images.front"]
    arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    Image.fromarray(arr).save(os.path.join(OUT, "images", f"{i:05d}.png"))
    pairs.append({"img": f"images/{i:05d}.png", "bins": bins.flatten().tolist(),
                  "ep": ep_i, "start": int(start), "motion": round(motion, 1)})
    if (i + 1) % 200 == 0:
        print(f"  decoded {i + 1}/{len(sel)}", flush=True)

json.dump(pairs, open(os.path.join(OUT, "pairs.json"), "w"))
np.savez(os.path.join(OUT, "bin_stats.npz"), lo=lo, hi=hi, idx=idx,
         n_bins=N_BINS, waypoints=WAYPOINTS, horizon=HORIZON)
print(f"wrote {len(pairs)} pairs | answer length is always {WAYPOINTS * 6} tokens")
