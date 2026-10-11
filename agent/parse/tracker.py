"""
tracker.py - consumes Showdown protocol lines one at a time and maintains BattleState.

No feature engineering here. This only answers: "what is true about the battle right now?"
The extractor (extract.py) calls snapshot() at each |turn| and records labels.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import re

STAT_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
HAZARDS = {"Stealth Rock", "Spikes", "Toxic Spikes", "Sticky Web"}


# ---------- tokenizing helpers ----------

def split_line(line: str):
    """'|move|p1a: X|Earthquake|p2a: Y|[miss]' -> ('move', ['p1a: X', 'Earthquake', 'p2a: Y'], {'miss': ''})"""
    parts = line.split("|")[1:]  # first element is '' because lines start with '|'
    if not parts:
        return "", [], {}
    event, rest = parts[0], parts[1:]
    args, tags = [], {}
    for a in rest:
        if a.startswith("[") and "]" in a:
            k, _, v = a[1:].partition("]")
            tags[k] = v.strip()
        else:
            args.append(a)
    return event, args, tags


def parse_ident(ident: str):
    """'p1a: Beldum' -> ('p1', 'Beldum');  'p1: Alice' -> ('p1', 'Alice')"""
    pos, _, name = ident.partition(": ")
    return pos[:2], name


def parse_details(details: str):
    """'Garchomp, L80, M' -> ('Garchomp', 80). Level defaults to 100."""
    bits = [b.strip() for b in details.split(",")]
    level = 100
    for b in bits[1:]:
        if re.fullmatch(r"L\d+", b):
            level = int(b[1:])
    return bits[0], level


def parse_hp(hp_str: str):
    """'43/100' | '0 fnt' | '43/100 par' -> (fraction, status or None)"""
    cond, _, status = hp_str.partition(" ")
    if cond == "0":
        return 0.0, "fnt"
    cur, _, mx = cond.partition("/")
    frac = int(cur) / int(mx) if mx else 1.0
    return frac, (status or None)


# ---------- state ----------

@dataclass
class Mon:
    key: str                       # stable identity within the side (species at first sight)
    species: str
    level: int = 100
    hp: float = 1.0                # fraction 0..1 (spectator logs only give percentages)
    status: str | None = None
    fainted: bool = False
    revealed: bool = False         # has been sent out at least once
    moves: set = field(default_factory=set)   # moves seen used so far
    tera_type: str | None = None
    boosts: dict = field(default_factory=dict)

    def reset_volatile(self):
        self.boosts = {}


@dataclass
class Side:
    id: str
    player: str = ""
    rating: int | None = None
    mons: dict = field(default_factory=dict)   # key -> Mon
    nick: dict = field(default_factory=dict)   # nickname -> key
    active_key: str | None = None
    hazards: dict = field(default_factory=dict)  # name -> layers
    tera_used: bool = False

    @property
    def active(self) -> Mon | None:
        return self.mons.get(self.active_key) if self.active_key else None


class BattleState:
    def __init__(self):
        self.sides = {"p1": Side("p1"), "p2": Side("p2")}
        self.turn = 0
        self.weather: str | None = None
        self.terrain: str | None = None
        self.pseudo: set = set()       # Trick Room, Gravity, ...
        self.winner: str | None = None  # player name from |win|

    # ----- public API -----

    def apply(self, line: str):
        event, args, tags = split_line(line)
        if not event:
            return
        handler = getattr(self, "_on_" + event.lstrip("-"), None)
        if handler:
            handler(args, tags)

    def winner_side(self) -> str | None:
        for sid, s in self.sides.items():
            if s.player and s.player == self.winner:
                return sid
        return None

    def snapshot(self, me: str) -> dict:
        """Raw snapshot from `me`'s perspective. Store this, featurize later."""
        opp = "p2" if me == "p1" else "p1"

        def mon_dict(m: Mon | None):
            if m is None:
                return None
            return dict(key=m.key, species=m.species, level=m.level, hp=m.hp, status=m.status,
                        fainted=m.fainted, moves=sorted(m.moves),
                        tera_type=m.tera_type, boosts=dict(m.boosts))

        def side_dict(s: Side):
            return dict(
                active=mon_dict(s.active),
                bench=[mon_dict(m) for k, m in s.mons.items() if k != s.active_key],
                hazards=dict(s.hazards),
                tera_used=s.tera_used,
            )

        return dict(turn=self.turn, me=side_dict(self.sides[me]),
                    opp=side_dict(self.sides[opp]),
                    weather=self.weather, terrain=self.terrain,
                    pseudo=sorted(self.pseudo))

    # ----- internals -----

    def _find_or_create(self, side: Side, nick: str, details: str | None = None) -> Mon:
        if nick in side.nick:
            return side.mons[side.nick[nick]]
        species, level = parse_details(details) if details else (nick, 100)
        key = species
        if key not in side.mons:
            # team preview may list 'Urshifu-*' while the switch says 'Urshifu-Rapid-Strike'
            base = species.split("-")[0]
            wild = [k for k in side.mons if k.endswith("-*") and k.split("-")[0] == base]
            if wild:
                key = wild[0]
                side.mons[key].species = species
            else:
                side.mons[key] = Mon(key=key, species=species, level=level)
        side.nick[nick] = key
        side.mons[key].level = level
        return side.mons[key]

    def _mon(self, ident: str) -> Mon | None:
        sid, nick = parse_ident(ident)
        side = self.sides.get(sid)
        if side is None:
            return None
        return self._find_or_create(side, nick)

    # ----- event handlers (named _on_<event without leading '-'>) -----

    def _on_player(self, a, t):
        if len(a) >= 2:
            side = self.sides[a[0]]
            side.player = a[1]
            if len(a) >= 4 and a[3].isdigit():
                side.rating = int(a[3])

    def _on_poke(self, a, t):                      # team preview: |poke|p1|Garchomp, L80, M|
        side = self.sides[a[0]]
        species, level = parse_details(a[1])
        side.mons[species] = Mon(key=species, species=species, level=level)

    def _on_switch(self, a, t):
        sid, nick = parse_ident(a[0])
        side = self.sides[sid]
        # the outgoing mon loses its volatile state
        if side.active:
            side.active.reset_volatile()
        mon = self._find_or_create(side, nick, a[1])
        mon.revealed = True
        mon.hp, mon.status = parse_hp(a[2])
        mon.fainted = mon.hp == 0
        mon.reset_volatile()
        side.active_key = mon.key

    _on_drag = _on_switch

    def _on_detailschange(self, a, t):             # permanent forme change / mega
        m = self._mon(a[0])
        if m:
            m.species, _ = parse_details(a[1])

    def _on_move(self, a, t):
        # Only count it as a revealed move if the player *chose* it: no [from] tag.
        # (Sleep Talk calls, locked moves, Dancer etc. carry [from].)
        m = self._mon(a[0])
        if m and "from" not in t and a[1] not in ("Struggle", "Recharge"):
            m.moves.add(a[1])

    def _on_damage(self, a, t):
        m = self._mon(a[0])
        if m:
            m.hp, st = parse_hp(a[1])
            if st == "fnt":
                m.fainted, m.status = True, None
            elif st:
                m.status = st

    _on_heal = _on_damage
    _on_sethp = _on_damage

    def _on_faint(self, a, t):
        m = self._mon(a[0])
        if m:
            m.hp, m.fainted, m.status = 0.0, True, None

    def _on_status(self, a, t):
        m = self._mon(a[0])
        if m:
            m.status = a[1]

    def _on_curestatus(self, a, t):
        m = self._mon(a[0])
        if m:
            m.status = None

    def _on_boost(self, a, t):
        m = self._mon(a[0])
        if m and a[1] in STAT_KEYS:
            m.boosts[a[1]] = max(-6, min(6, m.boosts.get(a[1], 0) + int(a[2])))

    def _on_unboost(self, a, t):
        m = self._mon(a[0])
        if m and a[1] in STAT_KEYS:
            m.boosts[a[1]] = max(-6, min(6, m.boosts.get(a[1], 0) - int(a[2])))

    def _on_setboost(self, a, t):
        m = self._mon(a[0])
        if m and a[1] in STAT_KEYS:
            m.boosts[a[1]] = int(a[2])

    def _on_clearallboost(self, a, t):
        for s in self.sides.values():
            for m in s.mons.values():
                m.reset_volatile()

    def _on_weather(self, a, t):
        if "upkeep" in t:
            return
        self.weather = None if a[0] == "none" else a[0]

    def _on_fieldstart(self, a, t):
        name = a[0].replace("move: ", "")
        if name.endswith("Terrain"):
            self.terrain = name
        else:
            self.pseudo.add(name)

    def _on_fieldend(self, a, t):
        name = a[0].replace("move: ", "")
        if name.endswith("Terrain"):
            self.terrain = None
        else:
            self.pseudo.discard(name)

    def _on_sidestart(self, a, t):
        sid, _ = parse_ident(a[0])
        name = a[1].replace("move: ", "")
        h = self.sides[sid].hazards
        h[name] = h.get(name, 0) + 1 if name in ("Spikes", "Toxic Spikes") else 1

    def _on_sideend(self, a, t):
        sid, _ = parse_ident(a[0])
        self.sides[sid].hazards.pop(a[1].replace("move: ", ""), None)

    def _on_terastallize(self, a, t):
        sid, _ = parse_ident(a[0])
        m = self._mon(a[0])
        if m:
            m.tera_type = a[1]
        self.sides[sid].tera_used = True

    def _on_turn(self, a, t):
        self.turn = int(a[0])

    def _on_win(self, a, t):
        self.winner = a[0]


# ---------- smoke test ----------

if __name__ == "__main__":
    import sys
    log = open(sys.argv[1]).read() if len(sys.argv) > 1 else ""
    st = BattleState()
    for ln in log.splitlines():
        st.apply(ln)
    print("winner side:", st.winner_side(), "| turns:", st.turn)
    for sid, s in st.sides.items():
        print(sid, s.player, [(m.species, round(m.hp, 2), sorted(m.moves)) for m in s.mons.values()])