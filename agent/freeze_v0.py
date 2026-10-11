"""
freeze_v0.py - retrain WITHOUT cand_revealed at a fixed depth and save the model the live
bot will use. Run once; after that, treat the saved file as read-only (it is the baseline).

Usage:
    python freeze_v0.py features_5k.parquet beldum_v0.joblib [depth]
"""
from __future__ import annotations
import sys

import joblib
import numpy as np
import pandas as pd

from parse.features import feature_columns
from train import split_by_battle, fit, score, top1_hits, SEED

DROP = ["cand_revealed"]

if __name__ == "__main__":
    df = pd.read_parquet(sys.argv[1])
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    cols = [c for c in feature_columns(df) if c not in DROP]

    train, val, test = split_by_battle(df)
    tv = pd.concat([train, val], ignore_index=True)
    clf = fit(tv, cols, depth)

    # sanity check: should land near the 0.467 you got in the cand_revealed ablation
    rng = np.random.default_rng(SEED)
    test = score(clf, test, cols)
    print("offline test top-1:", round(top1_hits(test, "score", rng).mean(), 4))

    joblib.dump({"model": clf, "features": cols, "depth": depth,
                 "dropped": DROP, "n_train_rows": len(tv)}, sys.argv[2])
    print(f"saved {sys.argv[2]} | depth {depth} | {len(cols)} features | trained on {len(tv)} rows")