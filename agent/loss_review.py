"""
loss_review.py - why does Beldum lose? Compares wins vs losses from a --snapshots run.

Usage (use the files from ONE run; battle tags repeat across runs):
  python agent\\env\\loss_review.py --decisions results\\decisions_heuristic_XXXX.jsonl
         --games results\\games_heuristic_XXXX.csv --pokedex data\\pokedex.json --moves data\\moves.json

Definitions (rough, on purpose):
  bad matchup      the opponent's STAB hits my active for 2x+ AND my active's STAB hits for 1x or less
  stayed in        a bad matchup, a healthy bench mon was available, and Beldum clicked a move
  status loop      the same status move chosen 4 times in a row (e.g. Calm Mind spam)
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
from collections import Counter, defaultdict

from parse.features import state_features
from parse.static import Dex


def load(decisions_pat, games_pat):
    won = {}
    for f in glob.glob(games_pat):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                won[r["battle"]] = r["won"] == "1"
    recs = []
    for f in glob.glob(decisions_pat):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if "snapshot" in r and r["battle"] in won:
                    recs.append(r)
    return won, recs


def pct(a, b):
    return f"{100 * a / b:5.1f}%" if b else "   n/a"


def analyze(won, recs, dex):
    by_battle = defaultdict(list)
    for r in recs:
        by_battle[r["battle"]].append(r)

    stats = {True: Counter(), False: Counter()}
    moves = {True: Counter(), False: Counter()}
    examples = []

    for b, rs in by_battle.items():
        w = won[b]
        st = stats[w]
        st["games"] += 1
        last, streak = None, 0
        for r in sorted(rs, key=lambda x: x["turn"]):
            if r["forced"]:
                continue
            out = state_features(r["snapshot"], dex)
            if out is None:
                continue
            sf, ma, oa = out
            kind, name = r["chosen"]
            st["voluntary"] += 1
            st["switches"] += int(kind == "switch")
            if kind == "move":
                moves[w][name] += 1

            bad = sf["opp_stab_vs_me"] >= 2 and sf["me_stab_vs_opp"] <= 1
            switchable = any(not m["fainted"] for m in r["snapshot"]["me"]["bench"])
            st["bad_matchup"] += int(bad)
            if bad and switchable:
                st["bad_switchable"] += 1
                if kind == "move":
                    st["bad_stay"] += 1
                    if not w and len(examples) < 10:
                        examples.append((b, r["turn"], ma["species"], oa["species"], name))

            mv = dex.move(name) if kind == "move" else None
            is_status = bool(mv) and mv.get("category") == "Status"
            streak = streak + 1 if (kind, name) == last and is_status else 1
            last = (kind, name)
            if is_status and streak == 4:
                st["status_loops"] += 1
    return stats, moves, examples


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--games", required=True)
    ap.add_argument("--pokedex", required=True)
    ap.add_argument("--moves", required=True)
    a = ap.parse_args()

    dex = Dex(a.pokedex, a.moves)
    won, recs = load(a.decisions, a.games)
    if not recs:
        raise SystemExit("no decisions with snapshots matched; rerun run_eval.py with --snapshots")
    stats, moves, examples = analyze(won, recs, dex)

    print(f"{'':8}{'games':>6}{'voluntary':>10}{'switch%':>9}{'bad matchup':>13}"
          f"{'stayed in*':>12}{'status loops':>14}")
    for label, w in (("WINS", True), ("LOSSES", False)):
        s = stats[w]
        print(f"{label:8}{s['games']:>6}{s['voluntary']:>10}{pct(s['switches'], s['voluntary']):>9}"
              f"{pct(s['bad_matchup'], s['voluntary']):>13}"
              f"{pct(s['bad_stay'], s['bad_switchable']):>12}{s['status_loops']:>14}")
    print("* share of bad-matchup turns (with a healthy switch available) where Beldum clicked a move\n")

    for label, w in (("LOSSES", False), ("WINS", True)):
        top = ", ".join(f"{n} ({c})" for n, c in moves[w].most_common(8))
        print(f"most chosen moves in {label}: {top}")
    print("\nexamples of staying in a bad matchup, in losses:")
    for b, t, mine, theirs, mv in examples:
        print(f"  {b} turn {t}: {mine} vs {theirs} -> {mv}")