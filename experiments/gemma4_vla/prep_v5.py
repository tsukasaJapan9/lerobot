"""Continuous (image, state) -> 50x6 action chunk pairs for the flow-matching policy.

v1-v4 discretised the actions and generated them autoregressively; greedy decoding then always
picked the safest next token and the trajectory collapsed to a constant. pi0 keeps actions
continuous and denoises the whole chunk at once, so that failure mode does not exist. This just
re-exports the same windows (and the images already decoded for v3) as continuous arrays.

Normalisation is MEAN_STD, as in pi0/ACT.
"""
import glob, json, os

import numpy as np
import pandas as pd

ROOT = "/media/tsukasa/DATA/so101/pick_and_place"
HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.join(HERE, "out_v3")  # reuse the decoded frames
OUT = os.path.join(HERE, "out_v5")
HORIZON = 50

os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(f"{ROOT}/data/**/*.parquet", recursive=True))
df = pd.concat([pd.read_parquet(f) for f in files]).sort_values("index").reset_index(drop=True)
actions = np.stack(df["action"].to_numpy()).astype("float32")
states = np.stack(df["observation.state"].to_numpy()).astype("float32")

a_mean, a_std = actions.mean(0), actions.std(0) + 1e-6
s_mean, s_std = states.mean(0), states.std(0) + 1e-6
np.savez(os.path.join(OUT, "norm.npz"), a_mean=a_mean, a_std=a_std, s_mean=s_mean, s_std=s_std)

for split in ("pairs", "val"):
    pairs = json.load(open(os.path.join(V3, f"{split}.json")))
    chunks, st, imgs, meta = [], [], [], []
    for p in pairs:
        s = p["start"]
        chunks.append((actions[s : s + HORIZON] - a_mean) / a_std)  # [50,6]
        st.append((states[s] - s_mean) / s_std)  # [6]
        imgs.append(os.path.join(V3, p["img"]))
        meta.append({"ep": p["ep"], "start": s, "motion": p["motion"]})
    np.savez(os.path.join(OUT, f"{split}.npz"),
             actions=np.stack(chunks).astype("float32"),
             state=np.stack(st).astype("float32"))
    json.dump({"images": imgs, "meta": meta}, open(os.path.join(OUT, f"{split}.json"), "w"))
    print(f"{split}: {len(chunks)} pairs | actions {np.stack(chunks).shape} | state {np.stack(st).shape}")
