"""
team_check.py - how much training data does each member of your team have?

Usage:
    python team_check.py teams\\team.txt decisions_5k.parquet [max_rows]

Counts, per team species, the winner decisions where that species was the active mon.
Species with only a handful of decisions are ones the tree has barely seen play.
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter

import pandas as pd


def parse_team_species(text: str):
    """Species from a Showdown team export ('Nick (Species) (M) @ Item' / 'Species @ Item')."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        first = lines[0].split(" @ ")[0].strip()
        first = re.sub(r"\s*\((M|F)\)\s*$", "", first)
        m = re.search(r"\(([^()]+)\)\s*$", first)
        out.append((m.group(1) if m else first).strip())
    return out


if __name__ == "__main__":
    species = parse_team_species(open(sys.argv[1], encoding="utf-8").read())
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 150000
    dec = pd.read_parquet(sys.argv[2], columns=["is_winner", "state_json"])
    dec = dec[dec["is_winner"]].head(limit)

    seen = Counter()
    for s in dec["state_json"]:
        seen[json.loads(s)["me"]["active"]["species"]] += 1

    print(f"team ({len(species)}): {species}")
    print(f"scanned {len(dec)} winner decisions\n")
    for sp in species:
        n = sum(c for name, c in seen.items() if name == sp or name.startswith(sp + "-"))
        flag = "  <-- thin" if n < 1000 else ""
        print(f"  {sp:<28}{n:>8}{flag}")