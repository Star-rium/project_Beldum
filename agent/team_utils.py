"""
team_utils.py - normalize a Showdown team export before handing it to poke-env.

poke-env's team parser rejected exports containing gender, either as a separate
'Gender: M' line or as '(M)' after the name (it read '(M) @ Leftovers' as the item).
Gender has no effect on battle outcomes, so it is simply dropped and the server assigns one.
"""
from __future__ import annotations
import re


def clean_team(text: str) -> str:
    blocks = re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip())
    cleaned = []
    for block in blocks:
        lines = [ln.rstrip() for ln in block.strip().split("\n") if ln.strip()]
        if not lines:
            continue
        header = re.sub(r"\s*\((?:M|F)\)", "", lines[0])        # 'Kingambit (M) @ X' -> 'Kingambit @ X'
        body = [ln for ln in lines[1:] if not ln.strip().startswith("Gender:")]
        cleaned.append("\n".join([header] + body))
    return "\n\n".join(cleaned) + "\n"


if __name__ == "__main__":
    import sys
    print(clean_team(open(sys.argv[1], encoding="utf-8").read()))