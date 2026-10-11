"""
train.py - fit a decision tree that scores candidate actions, evaluate top-1 per DECISION.

Usage:
    python train.py features_5k.parquet [beldum_tree.joblib]

Rows are candidates (chosen = 0/1). The tree gives each candidate a probability of being
chosen; the bot's pick for a decision is the highest-scoring candidate. So the metric is
"did the top candidate match the player's real action", measured per decision, not per row.
"""
from __future__ import annotations
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from parse.features import feature_columns

SEED = 0
DEPTHS = [3, 5, 8, 12, 16]
MIN_LEAF = 50


# ---------- split ----------

def split_by_battle(df, seed=SEED):
    """70/15/15 by battle_id, so turns of one battle never straddle train and test."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tv_idx, te_idx = next(gss.split(df, groups=df["battle_id"].to_numpy()))
    tv, te = df.iloc[tv_idx], df.iloc[te_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    tr_idx, va_idx = next(gss2.split(tv, groups=tv["battle_id"].to_numpy()))
    return (tv.iloc[tr_idx].reset_index(drop=True),
            tv.iloc[va_idx].reset_index(drop=True),
            te.reset_index(drop=True))


# ---------- evaluation ----------

def top1_hits(d, col, rng):
    """Per decision: 1 if the highest-scoring candidate is the chosen one.
    Ties get random noise. Without it, idxmax would pick the first row of a tie, and
    moves come before switches in the table, which would bias the result."""
    tmp = d[["decision_id", "chosen"]].copy()
    tmp["s"] = d[col].to_numpy() + rng.random(len(d)) * 1e-9
    idx = tmp.groupby("decision_id")["s"].idxmax()
    return tmp.loc[idx].set_index("decision_id")["chosen"]


def decision_types(d):
    c = d[d["chosen"] == 1].set_index("decision_id")
    forced = c["is_forced"].astype(bool).to_numpy()
    kinds = np.where(forced, "forced " + c["forced_kind"].astype(str), "voluntary " + c["cand_type"])
    return pd.Series(kinds, index=c.index)


def move_rank_acc(d, col, rng):
    """Among decisions where the player clicked a move: did we pick the right MOVE,
    ignoring the move-vs-switch question."""
    m = d[d["cand_is_move"] == 1]
    m = m[m.groupby("decision_id")["chosen"].transform("sum") == 1]
    return top1_hits(m, col, rng).mean()


def fit(train, cols, depth):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=SEED)
    clf.fit(train[cols], train["chosen"])
    return clf


def score(clf, d, cols, col="score"):
    d[col] = clf.predict_proba(d[cols])[:, 1]
    return d


# ---------- main ----------

if __name__ == "__main__":
    df = pd.read_parquet(sys.argv[1])
    cols = feature_columns(df)
    rng = np.random.default_rng(SEED)

    train, val, test = split_by_battle(df)
    for name, d in (("train", train), ("val", val), ("test", test)):
        print(f"{name}: {d.battle_id.nunique()} battles, {d.decision_id.nunique()} decisions, {len(d)} rows")
    print(f"features: {len(cols)}\n")

    # 1) pick depth on validation
    print("depth sweep (top-1 per decision):")
    best_depth, best_val = None, -1
    for depth in DEPTHS:
        clf = fit(train, cols, depth)
        tr_acc = top1_hits(score(clf, train, cols), "score", rng).mean()
        va_acc = top1_hits(score(clf, val, cols), "score", rng).mean()
        print(f"  depth {depth:>2}: train {tr_acc:.4f} | val {va_acc:.4f}")
        if va_acc > best_val:
            best_depth, best_val = depth, va_acc
    print(f"-> using depth {best_depth}\n")

    # 2) final model on train+val, report once on test
    tv = pd.concat([train, val], ignore_index=True)
    clf = fit(tv, cols, best_depth)
    test = score(clf, test, cols)
    hits = top1_hits(test, "score", rng)

    types = decision_types(test)
    hits = hits.reindex(types.index)

    # baseline: always click the highest power_score move (forced rows: random)
    test["base"] = np.where(test["cand_is_move"] == 1, test["cand_power_score"], -1.0)
    base_hits = top1_hits(test, "base", rng).reindex(types.index)
    rand = (1 / test.groupby("decision_id").size()).reindex(types.index)

    tbl = pd.DataFrame({
        "n": types.value_counts(),
        "tree": hits.groupby(types).mean(),
        "max_power_move": base_hits.groupby(types).mean(),
        "random": rand.groupby(types).mean(),
    })
    tbl.loc["ALL"] = [len(types), hits.mean(), base_hits.mean(), rand.mean()]
    print("test set, top-1 per decision:")
    print(tbl.round(3).to_string(), "\n")

    test["base_move"] = np.where(test["cand_is_move"] == 1, test["cand_power_score"], -1.0)
    print("move ranking only (decisions where a move was clicked):")
    print(f"  tree {move_rank_acc(test, 'score', rng):.3f} | "
          f"max_power_move {move_rank_acc(test, 'base_move', rng):.3f}\n")

    # 3) what is the tree using?
    imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
    print("top 15 feature importances:")
    print(imp.head(15).round(4).to_string(), "\n")

    # 4) ablation: cand_revealed may not mean the same thing live as in training
    cols2 = [c for c in cols if c != "cand_revealed"]
    clf2 = fit(tv, cols2, best_depth)
    test = score(clf2, test, cols2, col="score2")
    print("without cand_revealed:")
    print(f"  overall top-1 {top1_hits(test, 'score2', rng).mean():.3f} "
          f"(with: {hits.mean():.3f}) | "
          f"move ranking {move_rank_acc(test, 'score2', rng):.3f}")

    if len(sys.argv) > 2:
        joblib.dump({"model": clf, "features": cols}, sys.argv[2])
        print("saved", sys.argv[2])