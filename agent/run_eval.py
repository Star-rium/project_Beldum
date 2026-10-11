"""
run_eval.py - play Beldum against a poke-env baseline and report win rate + behavior stats.

Start a local Showdown server first (node pokemon-showdown start --no-security), then:

  python run_eval.py --model beldum_v0.joblib --pokedex pokedex.json --moves moves.json ^
         --team teams\\team.txt --opponent random --n 20

Opponents: random | maxpower | heuristic. Both sides use the same team file, so results
measure decisions, not team strength. Half the games are challenged by each side.
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import math
import time
from pathlib import Path

from poke_env.player import RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer

from player import BeldumPlayer
from parse.static import Dex
from team_utils import clean_team

OPPONENTS = {"random": RandomPlayer, "maxpower": MaxBasePowerPlayer,
             "heuristic": SimpleHeuristicsPlayer}


def wilson(wins: int, n: int, z: float = 1.96):
    """95% Wilson interval for a win rate."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


async def play(args):
    dex = Dex(args.pokedex, args.moves)
    team = clean_team(Path(args.team).read_text(encoding="utf-8")) if args.team else None
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.opponent}{'_rule' if args.switch_rule else ''}_{time.strftime('%Y%m%d_%H%M%S')}"

    me = BeldumPlayer(
        battle_format=args.format, team=team, max_concurrent_battles=args.concurrent,
        model_path=args.model, dex=dex, switch_rule=args.switch_rule,
        log_path=out / f"decisions_{tag}.jsonl" if args.snapshots else None,
        log_snapshots=args.snapshots)
    opp = OPPONENTS[args.opponent](
        battle_format=args.format, team=team, max_concurrent_battles=args.concurrent)

    half = args.n // 2
    try:
        await asyncio.wait_for(me.battle_against(opp, n_battles=half), timeout=args.timeout)
        await asyncio.wait_for(opp.battle_against(me, n_battles=args.n - half), timeout=args.timeout)
    except asyncio.TimeoutError:
        print("TIMEOUT: reporting partial results")

    rows = []
    for b in me.battles.values():
        if getattr(b, "won", None) is None:        # unfinished
            continue
        st = me.stats.get(b.battle_tag, {})
        rows.append({"battle": b.battle_tag, "opponent": args.opponent, "won": int(b.won is True),
                     "turns": b.turn, **st})

    n = len(rows)
    wins = sum(r["won"] for r in rows)
    lo, hi = wilson(wins, n)
    dec = sum(r.get("decisions", 0) for r in rows) or 1
    vol = sum(r.get("voluntary", 0) for r in rows) or 1
    summary = {
        "opponent": args.opponent, "format": args.format, "model": args.model, "team": args.team,
        "switch_rule": args.switch_rule,
        "rule_switches": sum(r.get("rule_switches", 0) for r in rows),
        "games": n, "wins": wins, "win_rate": round(wins / n, 4) if n else None,
        "ci95": [round(lo, 4), round(hi, 4)],
        "avg_turns": round(sum(r["turns"] for r in rows) / n, 1) if n else None,
        "voluntary_switch_rate": round(sum(r.get("voluntary_switches", 0) for r in rows) / vol, 4),
        "fallback_rate": round(sum(r.get("fallbacks", 0) for r in rows) / dec, 4),
    }
    print(json.dumps(summary, indent=2))

    (out / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if rows:
        with (out / f"games_{tag}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pokedex", required=True)
    ap.add_argument("--moves", required=True)
    ap.add_argument("--team", default=None, help="Showdown-format team export (required for OU)")
    ap.add_argument("--opponent", choices=sorted(OPPONENTS), default="random")
    ap.add_argument("--format", default="gen9ou")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--concurrent", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per half")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--switch-rule", action="store_true",
                    help="baseline experiment: leave bad matchups for a clearly better teammate")
    ap.add_argument("--snapshots", action="store_true",
                    help="log per-decision snapshots (needed for the parity test)")
    asyncio.run(play(ap.parse_args()))