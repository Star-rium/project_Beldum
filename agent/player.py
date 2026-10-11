"""
beldum_player.py - a poke-env Player that scores every legal action with the decision tree
and plays the top one. Never crashes a battle: any failure falls back to a random legal move,
and every fallback is counted.
"""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from poke_env.player import Player

from adapter import build_snapshot
from parse.features import candidates


class BeldumPlayer(Player):
    def __init__(self, *args, model_path, dex, log_path=None, log_snapshots=False,
                 switch_rule=False, **kwargs):
        # set our attributes first: Player.__init__ starts listening immediately
        bundle = joblib.load(model_path)
        self.model, self.features = bundle["model"], bundle["features"]
        self.dex = dex
        self.rng = np.random.default_rng(0)
        self.log_path = Path(log_path) if log_path else None
        self.log_snapshots = log_snapshots
        self.switch_rule = switch_rule
        self.stats = {}          # battle_tag -> counters
        super().__init__(*args, **kwargs)

    # ----- poke-env hook -----

    def choose_move(self, battle):
        st = self.stats.setdefault(battle.battle_tag, {
            "decisions": 0, "voluntary": 0, "voluntary_switches": 0, "fallbacks": 0,
            "rule_switches": 0})
        st["decisions"] += 1
        order = None
        try:
            order = self._decide(battle, st)
        except Exception as e:            # a bug must never forfeit the game
            self._log({"battle": battle.battle_tag, "turn": getattr(battle, "turn", None),
                       "error": repr(e)})
        if order is None:
            st["fallbacks"] += 1
            return self.choose_random_move(battle)
        return order

    # ----- decision -----

    def _decide(self, battle, st):
        snap, by_key = build_snapshot(battle)
        if snap is None:
            return None
        forced = bool(getattr(battle, "force_switch", False))
        moves_by_id = {m.id: m for m in battle.available_moves}
        legal_switch = {id(m) for m in battle.available_switches}

        cands = candidates(snap, list(moves_by_id), self.dex, forced=forced)
        if not cands:
            return None
        # keep only options the server will actually accept
        cands = [c for c in cands
                 if (c["cand_type"] == "move" and not forced and c["cand_name"] in moves_by_id)
                 or (c["cand_type"] == "switch" and id(by_key.get(c["cand_key"])) in legal_switch)]
        if not cands:
            return None

        X = pd.DataFrame(cands).reindex(columns=self.features).fillna(0)
        p = self.model.predict_proba(X)[:, 1]
        i = int(np.argmax(p + self.rng.random(len(p)) * 1e-9))     # random tie-break, as in eval
        ch = cands[i]

        if self.switch_rule and not forced:
            alt = self._rule_switch(cands)
            if alt is not None:
                ch = alt
                st["rule_switches"] += 1

        if not forced:
            st["voluntary"] += 1
            st["voluntary_switches"] += int(ch["cand_type"] == "switch")

        if self.log_path:
            top = sorted(zip(p, cands), key=lambda t: -t[0])[:3]
            rec = {"battle": battle.battle_tag, "turn": snap["turn"], "forced": forced,
                   "n_cands": len(cands), "chosen": [ch["cand_type"], ch["cand_name"]],
                   "top3": [[c["cand_type"], c["cand_name"], round(float(s), 4)] for s, c in top]}
            if self.log_snapshots:
                rec["snapshot"] = snap
            self._log(rec)

        if ch["cand_type"] == "move":
            return self.create_order(moves_by_id[ch["cand_name"]])
        return self.create_order(by_key[ch["cand_key"]])

    @staticmethod
    def _rule_switch(cands):
        """Crude baseline, not a learned behavior: leave a bad matchup for a clearly better,
        healthy teammate. Bad matchup = opponent STAB hits us 2x+ and ours hits 1x or less."""
        c0 = cands[0]                      # shared state features are on every candidate
        if not (c0["opp_stab_vs_me"] >= 2 and c0["me_stab_vs_opp"] <= 1):
            return None
        sw = [c for c in cands if c["cand_type"] == "switch" and c["cand_hp"] >= 0.5]
        if not sw:
            return None
        best = min(sw, key=lambda c: (c["cand_opp_stab_vs_it"], -c["cand_stab_vs_opp"], -c["cand_hp"]))
        return best if best["cand_opp_stab_vs_it"] < c0["opp_stab_vs_me"] else None

    def _log(self, rec):
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")