"""
features.py - turns decision rows into one row per (decision, candidate action).

The core is candidates(snapshot, my_moves, dex, forced): training and live play must both
call it, so the model sees identical features in both. In training, my_moves comes from
final_moves_json; live, it comes from Showdown's request data.

Usage:
    python features.py pokedex.json moves.json decisions_5k.parquet features_5k.parquet
"""
from __future__ import annotations
import json
import sys
from collections import Counter

from parse.static import Dex, type_mult, to_id

STATUSES = ("brn", "par", "psn", "tox", "slp", "frz")
BOOST_STATS = ("atk", "def", "spa", "spd", "spe")
BASE_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
HAZARDS = {"sr": "Stealth Rock", "spikes": "Spikes", "tspikes": "Toxic Spikes", "web": "Sticky Web"}
WEATHERS = {"sun": "SunnyDay", "rain": "RainDance", "sand": "Sandstorm", "snow": "Snow"}
TERRAINS = {"electric": "Electric Terrain", "grassy": "Grassy Terrain",
            "misty": "Misty Terrain", "psychic": "Psychic Terrain"}
HAZARD_MOVES = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
REMOVAL_MOVES = {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
PROTECT_MOVES = {"protect", "detect", "kingsshield", "spikyshield", "banefulbunker",
                 "silktrap", "burningbulwark", "obstruct"}

# columns that identify a row or hold the label; everything else is a model feature
META = ["decision_id", "battle_id", "turn", "side", "is_winner", "is_forced", "forced_kind",
        "rating", "tera", "chosen", "cand_type", "cand_name", "cand_key"]


def feature_columns(df):
    return [c for c in df.columns if c not in META]


# ---------- helpers ----------

def best_mult(att_types, def_types) -> float:
    return max((type_mult(t, def_types) for t in att_types), default=1.0)


def eff_speed(mon: dict, dex: Dex) -> float:
    """Estimated speed with boosts and paralysis. Ignores items, abilities, Tailwind."""
    s = dex.est_speed(mon["species"], mon["level"]) or 0.0
    b = mon["boosts"].get("spe", 0)
    s *= (2 + b) / 2 if b >= 0 else 2 / (2 - b)
    if mon["status"] == "par":
        s *= 0.5
    return s


# ---------- shared (per-decision) features ----------

def state_features(snap: dict, dex: Dex):
    me, opp = snap["me"], snap["opp"]
    ma, oa = me["active"], opp["active"]
    if ma is None or oa is None:
        return None
    mt, ot = dex.types(ma["species"]), dex.types(oa["species"])
    ms, os_ = eff_speed(ma, dex), eff_speed(oa, dex)

    f = {
        "me_hp": ma["hp"], "opp_hp": oa["hp"],
        "me_speed": ms, "opp_speed": os_, "outspeeds": int(ms > os_),
        "me_stab_vs_opp": best_mult(mt, ot), "opp_stab_vs_me": best_mult(ot, mt),
        "me_alive": sum(not m["fainted"] for m in me["bench"]) + int(not ma["fainted"]),
        "opp_alive": sum(not m["fainted"] for m in opp["bench"]) + int(not oa["fainted"]),
        "me_tera_used": int(me["tera_used"]), "opp_tera_used": int(opp["tera_used"]),
        "trick_room": int("Trick Room" in snap["pseudo"]),
    }
    for s in STATUSES:
        f[f"me_{s}"] = int(ma["status"] == s)
        f[f"opp_{s}"] = int(oa["status"] == s)
    for b in BOOST_STATS:
        f[f"me_boost_{b}"] = ma["boosts"].get(b, 0)
        f[f"opp_boost_{b}"] = oa["boosts"].get(b, 0)
    mb, ob = dex.base_stats(ma["species"]), dex.base_stats(oa["species"])
    for k in BASE_KEYS:
        f[f"me_base_{k}"] = mb.get(k, 0)
        f[f"opp_base_{k}"] = ob.get(k, 0)
    for short, name in HAZARDS.items():
        f[f"me_hz_{short}"] = me["hazards"].get(name, 0)
        f[f"opp_hz_{short}"] = opp["hazards"].get(name, 0)
    for short, name in WEATHERS.items():
        f[f"weather_{short}"] = int(snap["weather"] == name)
    for short, name in TERRAINS.items():
        f[f"terrain_{short}"] = int(snap["terrain"] == name)
    return f, ma, oa


# ---------- per-candidate features ----------

def move_candidate(name: str, ma: dict, oa: dict, dex: Dex):
    mf = dex.move_features(name, ma["species"], oa["species"])
    mv = dex.move(name)
    if mf is None or mv is None:
        return None
    mid = to_id(name)
    return {
        "cand_is_move": 1,
        "cand_base_power": mf["base_power"],
        "cand_priority": mf["priority"],
        "cand_accuracy": mf["accuracy"],
        "cand_type_mult": mf["type_mult"],
        "cand_stab": mf["stab"],
        "cand_power_score": mf["power_score"],
        "cand_physical": int(mf["category"] == "Physical"),
        "cand_special": int(mf["category"] == "Special"),
        "cand_status_move": int(mf["category"] == "Status"),
        "cand_revealed": int(name in ma["moves"]),    # moves used so far; compute the same way live
        # status moves are otherwise indistinguishable (power 0), so flag what they do
        "cand_hazard": int(mid in HAZARD_MOVES),
        "cand_removal": int(mid in REMOVAL_MOVES),
        "cand_heal": int(bool(mv.get("heal"))),
        "cand_drain": int(bool(mv.get("drain"))),
        "cand_boost_self": int(bool(mv.get("boosts")) and mv.get("target") == "self"),
        "cand_inflicts_status": int(bool(mv.get("status")) and mv.get("target") != "self"),
        "cand_pivot": int(bool(mv.get("selfSwitch"))),
        "cand_protect": int(bool(mv.get("stallingMove")) or mid in PROTECT_MOVES),
    }


def switch_candidate(m: dict, opp_types: list, opp_speed: float, dex: Dex):
    t = dex.types(m["species"])
    bs = dex.base_stats(m["species"])
    f = {
        "cand_is_move": 0,
        "cand_hp": m["hp"],
        "cand_statused": int(m["status"] is not None),
        "cand_stab_vs_opp": best_mult(t, opp_types),       # how well it hits the opponent
        "cand_opp_stab_vs_it": best_mult(opp_types, t),    # how hard the opponent hits it
        "cand_outspeeds": int(eff_speed(m, dex) > opp_speed),
    }
    for k in BASE_KEYS:
        f[f"cand_base_{k}"] = bs.get(k, 0)
    return f


def candidates(snap: dict, my_moves: list, dex: Dex, forced: bool = False):
    """All legal-ish candidates for one decision, each merged with the shared state features."""
    sf = state_features(snap, dex)
    if sf is None:
        return None
    shared, ma, oa = sf
    ot = dex.types(oa["species"])
    out = []
    if not forced:                                         # forced switches can't use a move
        for name in my_moves:
            cf = move_candidate(name, ma, oa, dex)
            if cf:
                out.append({**shared, "cand_type": "move", "cand_name": name, "cand_key": "", **cf})
    for m in snap["me"]["bench"]:
        if m["fainted"]:
            continue
        cf = switch_candidate(m, ot, shared["opp_speed"], dex)
        out.append({**shared, "cand_type": "switch", "cand_name": m["species"],
                    "cand_key": m["key"], **cf})
    return out


# ---------- training-set builder ----------

def _match_switch(cands: list, action_id: str):
    """Chosen switch target. The log gives the species at switch time, which may be a
    different forme from the one in the snapshot (Ogerpon-Tera, Terapagos-Terastal)."""
    sw = [i for i, c in enumerate(cands) if c["cand_type"] == "switch"]
    for i in sw:
        if action_id in (cands[i]["cand_name"], cands[i]["cand_key"]):
            return i
    base = action_id.split("-")[0]
    near = [i for i in sw if cands[i]["cand_name"].split("-")[0] == base
            or cands[i]["cand_key"].split("-")[0] == base]
    return near[0] if len(near) == 1 else None


def build_features(dec, dex: Dex, only_winner=True, include_forced=True, max_decisions=None):
    import pandas as pd
    if only_winner:
        dec = dec[dec["is_winner"]]
    if not include_forced:
        dec = dec[~dec["is_forced"]]
    if max_decisions:
        dec = dec.head(max_decisions)

    rows, stats = [], Counter()
    for r in dec.itertuples():
        snap = json.loads(r.state_json)
        ma = snap["me"]["active"]
        my_moves = json.loads(r.final_moves_json).get(ma["key"], []) if ma else []
        cands = candidates(snap, my_moves, dex, forced=bool(r.is_forced))
        if cands is None:
            stats["dropped: no active mon"] += 1
            continue
        if r.action_type == "move":
            idx = next((i for i, c in enumerate(cands)
                        if c["cand_type"] == "move" and c["cand_name"] == r.action_id), None)
        else:
            idx = _match_switch(cands, r.action_id)
        if idx is None:
            stats["dropped: chosen action not a candidate"] += 1
            continue
        stats["decisions kept"] += 1
        for i, c in enumerate(cands):
            c.update(decision_id=r.Index, battle_id=r.battle_id, turn=r.turn, side=r.side,
                     is_winner=bool(r.is_winner), is_forced=bool(r.is_forced),
                     forced_kind=r.forced_kind, rating=r.rating, tera=bool(r.tera),
                     chosen=int(i == idx))
            rows.append(c)
    df = pd.DataFrame(rows).fillna(0)       # move-only / switch-only columns become 0
    return df, stats


if __name__ == "__main__":
    import pandas as pd
    dex = Dex(sys.argv[1], sys.argv[2])
    dec = pd.read_parquet(sys.argv[3])
    df, stats = build_features(dec, dex)
    for k, v in stats.items():
        print(f"{k}: {v}")
    print("rows:", len(df), "| feature columns:", len(feature_columns(df)))
    print("candidates per decision:", round(len(df) / df.decision_id.nunique(), 2))
    print("every decision has exactly one chosen:",
          bool((df.groupby("decision_id").chosen.sum() == 1).all()))

    # naive baseline for the tree to beat: always click the highest power_score move
    mv = df[df.cand_is_move == 1]
    best = mv.loc[mv.groupby("decision_id").cand_power_score.idxmax()]
    n_dec = df.decision_id.nunique()
    print("baseline (max power_score move) top-1 accuracy:", round(best.chosen.sum() / n_dec, 4))

    if len(sys.argv) > 4:
        df.to_parquet(sys.argv[4])
        print("wrote", sys.argv[4])
