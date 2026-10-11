"""
adapter.py - builds, from a live poke-env battle, the SAME snapshot dict that tracker.py
produces from a log. features.candidates() then works unchanged in training and live play.

This is the most fragile file in the harness: it depends on poke-env attribute names and on
how poke-env stores hazards/weather/fields. Everything is read defensively (getattr), and the
parity test is what proves it matches the tracker. Check names against your installed version.
"""
from __future__ import annotations

from parse.static import to_id

TEAM_SIZE = 6
STATUS_MAP = {"BRN": "brn", "PAR": "par", "PSN": "psn", "TOX": "tox", "SLP": "slp", "FRZ": "frz"}
WEATHER_MAP = {"sunnyday": "SunnyDay", "raindance": "RainDance",
               "sandstorm": "Sandstorm", "snow": "Snow"}
# poke-env SideCondition name -> (name used in training snapshots, max layers)
HAZARDS = {"STEALTH_ROCK": ("Stealth Rock", 1), "SPIKES": ("Spikes", 3),
           "TOXIC_SPIKES": ("Toxic Spikes", 2), "STICKY_WEB": ("Sticky Web", 1)}


def _name(x) -> str:
    return getattr(x, "name", str(x))


def mon_dict(mon, key: str) -> dict:
    fainted = bool(getattr(mon, "fainted", False))
    status = None
    if not fainted and getattr(mon, "status", None) is not None:
        status = STATUS_MAP.get(_name(mon.status))
    return {
        "key": key,
        "species": mon.species,
        "level": int(getattr(mon, "level", 100) or 100),
        "hp": 0.0 if fainted else float(mon.current_hp_fraction),
        "status": status,
        "fainted": fainted,
        "moves": sorted(getattr(mon, "moves", {}) or {}),
        "tera_type": None,
        "boosts": dict(getattr(mon, "boosts", {}) or {}),
    }


def _unique(base: str, used: set) -> str:
    k, i = base, 2
    while k in used:
        k, i = f"{base}#{i}", i + 1
    used.add(k)
    return k


def hazards(conds) -> dict:
    out = {}
    for sc, val in (conds or {}).items():
        spec = HAZARDS.get(_name(sc))
        if spec:
            name, cap = spec
            out[name] = min(int(val), cap) if cap > 1 else 1
    return out


def tera_used(mons) -> bool:
    return any(bool(getattr(m, "is_terastallized", False)) or bool(getattr(m, "terastallized", False))
               for m in mons)


def field_state(battle):
    weather = None
    w = getattr(battle, "weather", None)
    if isinstance(w, dict):
        w = next(iter(w), None)
    if w is not None:
        weather = WEATHER_MAP.get(to_id(_name(w)))
    terrain, pseudo = None, []
    fields = getattr(battle, "fields", None) or {}
    for f in (fields.keys() if isinstance(fields, dict) else fields):
        name = _name(f).replace("_", " ").title()
        if name.endswith("Terrain"):
            terrain = name
        else:
            pseudo.append(name)
    return weather, terrain, pseudo


def build_snapshot(battle):
    """Returns (snapshot, by_key) where by_key maps each of OUR mons' keys to the poke-env
    Pokemon object, so a chosen switch candidate can be turned back into an order.
    Returns (None, {}) if either active mon is missing."""
    me_active = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon
    if me_active is None or opp_active is None:
        return None, {}

    used, by_key, key_of = set(), {}, {}
    for mon in battle.team.values():
        k = _unique(mon.species, used)
        by_key[k] = mon
        key_of[id(mon)] = k

    me_bench = [mon_dict(m, key_of[id(m)]) for m in battle.team.values() if m is not me_active]

    # The opponent's full team is known in training (team preview). Live, poke-env may only
    # expose revealed mons, so pad with unknown placeholders to keep alive-counts comparable.
    ukeys = set()
    opp_bench = [mon_dict(m, _unique(m.species, ukeys))
                 for m in battle.opponent_team.values() if m is not opp_active]
    while len(opp_bench) < TEAM_SIZE - 1:
        opp_bench.append({"key": f"unknown-{len(opp_bench)}", "species": "", "level": 100,
                          "hp": 1.0, "status": None, "fainted": False, "moves": [],
                          "tera_type": None, "boosts": {}})

    weather, terrain, pseudo = field_state(battle)
    snap = {
        "turn": int(getattr(battle, "turn", 0)),
        "me": {"active": mon_dict(me_active, key_of[id(me_active)]),
               "bench": me_bench,
               "hazards": hazards(getattr(battle, "side_conditions", {})),
               "tera_used": tera_used(battle.team.values())},
        "opp": {"active": mon_dict(opp_active, _unique(opp_active.species, ukeys)),
                "bench": opp_bench,
                "hazards": hazards(getattr(battle, "opponent_side_conditions", {})),
                "tera_used": tera_used(battle.opponent_team.values())},
        "weather": weather, "terrain": terrain, "pseudo": sorted(pseudo),
    }
    return snap, by_key