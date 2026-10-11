"""
static.py - static game knowledge for the featurizer.

Needs two files from Showdown (save them next to this file, e.g. beldum/static_data/):
    https://play.pokemonshowdown.com/data/pokedex.json
    https://play.pokemonshowdown.com/data/moves.json

Usage:
    python static.py pokedex.json moves.json            # quick smoke test
    python static.py pokedex.json moves.json decisions_5k.parquet   # coverage check
"""
from __future__ import annotations
import json
import math
import re

# ---------- type chart (attacker -> {defender: multiplier}; omitted = 1x) ----------
_C = {
    "Normal":   {"Rock": .5, "Steel": .5, "Ghost": 0},
    "Fire":     {"Fire": .5, "Water": .5, "Grass": 2, "Ice": 2, "Bug": 2, "Rock": .5, "Dragon": .5, "Steel": 2},
    "Water":    {"Fire": 2, "Water": .5, "Grass": .5, "Ground": 2, "Rock": 2, "Dragon": .5},
    "Electric": {"Water": 2, "Electric": .5, "Grass": .5, "Ground": 0, "Flying": 2, "Dragon": .5},
    "Grass":    {"Fire": .5, "Water": 2, "Grass": .5, "Poison": .5, "Ground": 2, "Flying": .5,
                 "Bug": .5, "Rock": 2, "Dragon": .5, "Steel": .5},
    "Ice":      {"Fire": .5, "Water": .5, "Grass": 2, "Ice": .5, "Ground": 2, "Flying": 2,
                 "Dragon": 2, "Steel": .5},
    "Fighting": {"Normal": 2, "Ice": 2, "Poison": .5, "Flying": .5, "Psychic": .5, "Bug": .5,
                 "Rock": 2, "Ghost": 0, "Dark": 2, "Steel": 2, "Fairy": .5},
    "Poison":   {"Grass": 2, "Poison": .5, "Ground": .5, "Rock": .5, "Ghost": .5, "Steel": 0, "Fairy": 2},
    "Ground":   {"Fire": 2, "Electric": 2, "Grass": .5, "Poison": 2, "Flying": 0, "Bug": .5,
                 "Rock": 2, "Steel": 2},
    "Flying":   {"Electric": .5, "Grass": 2, "Fighting": 2, "Bug": 2, "Rock": .5, "Steel": .5},
    "Psychic":  {"Fighting": 2, "Poison": 2, "Psychic": .5, "Dark": 0, "Steel": .5},
    "Bug":      {"Fire": .5, "Grass": 2, "Fighting": .5, "Poison": .5, "Flying": .5, "Psychic": 2,
                 "Ghost": .5, "Dark": 2, "Steel": .5, "Fairy": .5},
    "Rock":     {"Fire": 2, "Ice": 2, "Fighting": .5, "Ground": .5, "Flying": 2, "Bug": 2, "Steel": .5},
    "Ghost":    {"Normal": 0, "Psychic": 2, "Ghost": 2, "Dark": .5},
    "Dragon":   {"Dragon": 2, "Steel": .5, "Fairy": 0},
    "Dark":     {"Fighting": .5, "Psychic": 2, "Ghost": 2, "Dark": .5, "Fairy": .5},
    "Steel":    {"Fire": .5, "Water": .5, "Electric": .5, "Ice": 2, "Rock": 2, "Steel": .5, "Fairy": 2},
    "Fairy":    {"Fire": .5, "Fighting": 2, "Poison": .5, "Dragon": 2, "Dark": 2, "Steel": .5},
}
TYPES = list(_C)


def to_id(name: str) -> str:
    """Showdown's id normalization: 'Slowking-Galar' -> 'slowkinggalar'."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def type_mult(atk_type: str, def_types: list[str]) -> float:
    m = 1.0
    for d in def_types:
        m *= _C.get(atk_type, {}).get(d, 1.0)
    return m


class Dex:
    def __init__(self, pokedex_path: str, moves_path: str):
        with open(pokedex_path, encoding="utf-8") as f:
            self.pokedex = json.load(f)
        with open(moves_path, encoding="utf-8") as f:
            self.moves = json.load(f)

    # ----- lookups (return None when unknown; never guess silently) -----

    def species(self, name: str) -> dict | None:
        d = self.pokedex.get(to_id(name))
        if d is None and "-" in name:                       # 'Urshifu-*' or unseen forme
            d = self.pokedex.get(to_id(name.split("-")[0]))
        # cosmetic formes (Alcremie-Mint-Cream, Minior-Blue...) are stubs with no types or
        # stats; the real data sits on their baseSpecies entry
        for _ in range(2):
            if d is not None and "types" not in d and d.get("baseSpecies"):
                d = self.pokedex.get(to_id(d["baseSpecies"]))
        return d

    def move(self, name: str) -> dict | None:
        return self.moves.get(to_id(name))

    def types(self, species: str) -> list[str]:
        d = self.species(species)
        return list(d["types"]) if d else []

    def base_stats(self, species: str) -> dict:
        d = self.species(species)
        return dict(d["baseStats"]) if d else {}

    # ----- derived quantities -----

    def est_speed(self, species: str, level: int = 100) -> float | None:
        """Assumes 31 IV, 252 EV, neutral nature. Real value is hidden in spectator logs."""
        b = self.base_stats(species).get("spe")
        if b is None:
            return None
        return math.floor((2 * b + 31 + 63) * level / 100) + 5

    def move_features(self, move_name: str, user: str, target: str) -> dict | None:
        """Per-(move, matchup) numbers for the featurizer. Ignores abilities, items, Tera."""
        mv = self.move(move_name)
        if mv is None:
            return None
        acc = mv.get("accuracy", 100)
        acc = 1.0 if acc is True else acc / 100
        bp = mv.get("basePower", 0)
        mtype = mv["type"]
        mult = type_mult(mtype, self.types(target)) if mv["category"] != "Status" else 1.0
        stab = 1.5 if mtype in self.types(user) else 1.0
        return dict(
            base_power=bp,
            category=mv["category"],                       # Physical / Special / Status
            move_type=mtype,
            priority=mv.get("priority", 0),
            accuracy=acc,
            type_mult=mult,
            stab=stab,
            power_score=bp * mult * stab * acc,            # crude damage proxy, not a calc
        )


# ---------- smoke test + coverage check ----------

def coverage(dex: Dex, parquet_path: str, n: int = 20000):
    """Which species/moves in YOUR decisions file are missing from the Dex?"""
    import pandas as pd
    df = pd.read_parquet(parquet_path, columns=["action_type", "action_id", "state_json"]).head(n)
    species, moves = set(), set()
    for s in df["state_json"]:
        st = json.loads(s)
        for side in ("me", "opp"):
            mons = ([st[side]["active"]] if st[side]["active"] else []) + st[side]["bench"]
            for m in mons:
                species.add(m["species"])
                moves.update(m["moves"])
    moves.update(df.loc[df.action_type == "move", "action_id"])
    species.update(df.loc[df.action_type == "switch", "action_id"])
    miss_s = sorted(x for x in species if dex.species(x) is None)
    miss_m = sorted(x for x in moves if dex.move(x) is None)
    print(f"species: {len(species)} seen, {len(miss_s)} unresolved -> {miss_s[:20]}")
    print(f"moves:   {len(moves)} seen, {len(miss_m)} unresolved -> {miss_m[:20]}")


if __name__ == "__main__":
    import sys
    dex = Dex(sys.argv[1], sys.argv[2])
    print("Kingambit:", dex.types("Kingambit"), dex.base_stats("Kingambit"))
    print("Ogerpon-Wellspring:", dex.types("Ogerpon-Wellspring"))
    print("est speed Great Tusk L80:", dex.est_speed("Great Tusk", 80))
    print("Moonblast Enamorus->Kingambit:", dex.move_features("Moonblast", "Enamorus", "Kingambit"))
    print("Earthquake Roaring Moon->Heatran:", dex.move_features("Earthquake", "Roaring Moon", "Heatran"))
    if len(sys.argv) > 3:
        coverage(dex, sys.argv[3])