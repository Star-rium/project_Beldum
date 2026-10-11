"""
learning_curve.py - would more training data help?

Trains on 10/25/50/100% of the TRAIN battles and scores each model on the same validation
set. If accuracy is still rising at 100%, scaling up is worth it; if it has flattened, the
limit is the model or the features, not data volume.

Optionally also reports accuracy per team member (are thin species decided worse?).

Usage:
  python agent\\env\\learning_curve.py data\\features_5k.parquet
  python agent\\env\\learning_curve.py data\\features_5k.parquet --decisions data\\decisions_5k.parquet --team agent\\agent\\teams\\test_team.txt
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import pandas as pd

from parse.features import feature_columns
from team_check import parse_team_species
from train import split_by_battle, fit, score, top1_hits, SEED

FRACTIONS = [0.1, 0.25, 0.5, 1.0]
DEPTHS = [12, 16]
DROP = ["cand_revealed"]            # matches the frozen v0 feature set


def subsample(train, frac, rng):
    if frac >= 1:
        return train
    battles = train["battle_id"].unique()
    keep = rng.choice(battles, size=max(1, int(len(battles) * frac)), replace=False)
    return train[train["battle_id"].isin(keep)]


def curve(train, val, cols, rng):
    print(f"{'train share':>12}{'battles':>9}{'decisions':>11}" + "".join(f"{'depth ' + str(d):>10}" for d in DEPTHS))
    for frac in FRACTIONS:
        sub = subsample(train, frac, rng)
        accs = []
        for depth in DEPTHS:
            clf = fit(sub, cols, depth)
            v = score(clf, val.copy(), cols)
            accs.append(top1_hits(v, "score", rng).mean())
        print(f"{frac:>12.0%}{sub.battle_id.nunique():>9}{sub.decision_id.nunique():>11}"
              + "".join(f"{a:>10.4f}" for a in accs))


def per_species(train, val, test, cols, decisions_path, team_path, rng, depth=16):
    team = parse_team_species(open(team_path, encoding="utf-8").read())
    sj = pd.read_parquet(decisions_path, columns=["state_json"])["state_json"]
    held = pd.concat([val, test], ignore_index=True)           # neither was trained on
    clf = fit(train, cols, depth)
    held = score(clf, held, cols)
    hits = top1_hits(held, "score", rng)
    species = pd.Series({i: json.loads(sj.iloc[i])["me"]["active"]["species"] for i in hits.index})
    print(f"\ntop-1 by active species (depth {depth}, held-out val+test; all decisions: {hits.mean():.3f})")
    print(f"{'species':<16}{'decisions':>10}{'top-1':>8}")
    for sp in team:
        m = species.map(lambda s: s == sp or s.startswith(sp + "-"))
        n = int(m.sum())
        note = "  (too few to judge)" if n < 100 else ""
        print(f"{sp:<16}{n:>10}{hits[m.index[m]].mean() if n else float('nan'):>8.3f}{note}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--decisions", default=None)
    ap.add_argument("--team", default=None)
    a = ap.parse_args()

    df = pd.read_parquet(a.features)
    cols = [c for c in feature_columns(df) if c not in DROP]
    train, val, test = split_by_battle(df)
    rng = np.random.default_rng(SEED)

    curve(train, val, cols, rng)
    print("\nvalidation noise is about +/-0.3 points (21k decisions), so ignore smaller differences")
    if a.decisions and a.team:
        per_species(train, val, test, cols, a.decisions, a.team, rng)