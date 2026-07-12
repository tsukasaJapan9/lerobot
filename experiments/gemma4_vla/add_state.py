"""Add the robot's current joint state to each (image -> trajectory) pair.

v3 fed the model an image and nothing else, so it had to infer the arm's joint angles from a
top-down photo, and a single frame cannot tell "descending" from "ascending". Under that ambiguity
the loss-minimising answer is the average over possible futures, which is exactly the flat
trajectory it produced. Every real VLA (ACT, pi0, SmolVLA) gets proprioceptive state alongside the
image; this restores that input.

State is binned with the *action* min/max so a bin id means the same joint angle whether it appears
in the state prefix or in the predicted trajectory.
"""
import glob, json, os

import numpy as np
import pandas as pd

ROOT = "/media/tsukasa/DATA/so101/pick_and_place"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_v3")
N_BINS = 256

files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("index").reset_index(drop=True)
state = np.stack(df["observation.state"].to_numpy()).astype("float32")  # [N,6]

s = np.load(os.path.join(OUT, "bin_stats.npz"))
lo, hi = s["lo"], s["hi"]
span = np.maximum(hi - lo, 1e-6)

for name in ("pairs.json", "val.json"):
    path = os.path.join(OUT, name)
    pairs = json.load(open(path))
    for p in pairs:
        v = state[p["start"]]
        p["state_bins"] = np.clip(((v - lo) / span * (N_BINS - 1)).round(), 0, N_BINS - 1).astype(int).tolist()
    json.dump(pairs, open(path, "w"))
    print(f"{name}: added state_bins to {len(pairs)} pairs")

# sanity: the first waypoint of the trajectory should sit close to the current state
p = json.load(open(os.path.join(OUT, "pairs.json")))[0]
print("state bins     :", p["state_bins"])
print("first waypoint :", p["bins"][:6])
