"""
extract.py - replays a log through BattleState and emits one row per decision.

Phases within a battle:
    pre     before |turn|1          -> leads, not decisions
    choice  |turn|N .. |upkeep|     -> each side's first un-tagged move/switch is its choice
    post    |upkeep| .. next |turn| -> faint replacements (forced)

Row = (battle, turn, side) + label. Mid-turn pivots (Volt Switch, Chilly Reception...)
and post-faint replacements are kept but flagged with is_forced / forced_kind, so you can
train on them separately or drop them.
"""
from __future__ import annotations
import json

try:
    from tracker import BattleState, split_line, parse_details
except ImportError:  # when imported as part of a package
    from .tracker import BattleState, split_line, parse_details

NOT_A_CHOICE = {"Struggle", "Recharge"}


def extract_battle(battle_id: str, log: str):
    st = BattleState()
    rows = []
    phase = "pre"
    turn_snap = {}                       # state at the moment |turn|N appeared
    acted = set()                        # sides that already made their choice this turn
    tera = {"p1": False, "p2": False}

    def add(side, snap, action_type, action_id, forced=False, kind="", tera_flag=False):
        rows.append(dict(
            battle_id=battle_id, turn=st.turn, side=side,
            is_forced=forced, forced_kind=kind,
            action_type=action_type, action_id=action_id, tera=tera_flag,
            state_json=json.dumps(snap),
        ))

    for line in log.splitlines():
        event, args, tags = split_line(line)

        if event == "turn":
            st.apply(line)
            phase = "choice"
            turn_snap = {s: st.snapshot(s) for s in ("p1", "p2")}
            acted = set()
            tera = {"p1": False, "p2": False}
            continue

        if phase == "choice":
            if event == "upkeep":
                phase = "post"

            elif event == "-terastallize":
                tera[args[0][:2]] = True

            elif event == "move" and "from" not in tags and args[1] not in NOT_A_CHOICE:
                side = args[0][:2]
                if side not in acted:
                    add(side, turn_snap[side], "move", args[1], tera_flag=tera[side])
                    acted.add(side)

            elif event == "switch":
                side = args[0][:2]
                species = parse_details(args[1])[0]
                if "from" in tags or side in acted:
                    # pivot (U-turn, Volt Switch, Chilly Reception...): state is mid-turn,
                    # so snapshot NOW, before this switch is applied
                    add(side, st.snapshot(side), "switch", species, forced=True, kind="pivot")
                else:
                    add(side, turn_snap[side], "switch", species)
                    acted.add(side)

        elif phase == "post" and event == "switch":
            # replacement after a faint; active slot still holds the fainted mon here
            side = args[0][:2]
            add(side, st.snapshot(side), "switch", parse_details(args[1])[0],
                forced=True, kind="replace")

        st.apply(line)

    winner = st.winner_side()

    # Second-pass info: every move each side's mons revealed over the WHOLE battle.
    # Used as the candidate move list so a first-use move is still a candidate.
    # Only the acting side's team is attached (the opponent's unrevealed moves stay hidden).
    final_moves = {
        sid: json.dumps({m.key: sorted(m.moves) for m in side.mons.values()})
        for sid, side in st.sides.items()
    }
    for r in rows:
        r["is_winner"] = (r["side"] == winner)
        r["final_moves_json"] = final_moves[r["side"]]
    meta = dict(battle_id=battle_id, winner_side=winner, n_turns=st.turn,
                p1=st.sides["p1"].player, p2=st.sides["p2"].player)
    return meta, rows


def run(src, dst, format_id="gen9ou", min_rating=None, limit=None, batch_size=500):
    """Stream a battles parquet -> a decisions parquet. Develop with limit=5000."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(src)
    writer, n = None, 0
    for batch in pf.iter_batches(batch_size=batch_size,
                                 columns=["id", "formatid", "rating", "log"]):
        out = []
        for r in batch.to_pylist():
            if r["formatid"] != format_id or not r["log"]:
                continue
            if min_rating is not None and (r["rating"] or 0) < min_rating:
                continue
            meta, rows = extract_battle(r["id"], r["log"])
            if meta["winner_side"] is None:      # unfinished / forfeited-before-start
                continue
            rating = float(r["rating"]) if r["rating"] is not None else float("nan")
            for x in rows:
                x["rating"] = rating
            out.extend(rows)
            n += 1
            if limit and n >= limit:
                break
        if out:
            tbl = pa.Table.from_pylist(out)
            writer = writer or pq.ParquetWriter(dst, tbl.schema)
            writer.write_table(tbl)
        if limit and n >= limit:
            break
    if writer:
        writer.close()
    print(f"battles processed: {n}")


if __name__ == "__main__":
    import sys
    from collections import Counter
    log = open(sys.argv[1], encoding="utf-8").read()
    meta, rows = extract_battle("test", log)
    print(meta)
    c = Counter((r["side"], r["forced_kind"] or "voluntary") for r in rows)
    for k in sorted(c):
        print(k, c[k])
    print("total rows:", len(rows))
    for r in rows[:8]:
        print(r["turn"], r["side"], r["action_type"], r["action_id"],
              "forced=" + (r["forced_kind"] or "no"), "tera=" + str(r["tera"]))