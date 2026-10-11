from parse.extract import run
from parse.static import Dex
import pandas as pd
import json
import os
from collections import Counter

# run the extraction process with specified parameters
# run(
#     src="data/20920OU_part1.parquet",
#     dst="data/decisions_5k.parquet",
#     format_id="gen9ou",
#     limit=5000,
#     min_rating=1300,
# )

# # test the processed data output
# df = pd.read_parquet("data/decisions_5k.parquet")
# mv = df[df.action_type == "move"].copy()
# print(df.shape, df["battle_id"].nunique())
# print(df.groupby(["is_forced", "forced_kind"]).size())
# print(df["action_type"].value_counts())

# per_battle = df.groupby("battle_id").size()
# print(per_battle.describe())

# print(os.path.getsize("data/20920OU_decisions.parquet") / 1e6, "MB")

# turns = df.groupby("battle_id")["turn"].max()
# print((per_battle / turns).describe())        # rows per turn, expect ~1.5-2.5

# print(df["is_winner"].mean())                  # expect close to 0.5
# print(df["rating"].describe())                 # to choose a rating cutoff
# print(df[df.action_type == "move"]["tera"].mean())

# from parse.static import type_mult
# assert type_mult("Ground", ["Flying"]) == 0
# assert type_mult("Electric", ["Ground"]) == 0
# assert type_mult("Fairy", ["Dragon"]) == 2
# assert type_mult("Dragon", ["Fairy"]) == 0
# assert type_mult("Poison", ["Steel"]) == 0
# assert type_mult("Ice", ["Dragon", "Flying"]) == 4
# assert type_mult("Fighting", ["Ghost"]) == 0
# assert type_mult("Steel", ["Fairy"]) == 2

# def in_final(r):
#     act = json.loads(r.state_json)["me"]["active"]
#     return r.action_id in json.loads(r.final_moves_json).get(act["key"], [])

# ok = mv.apply(in_final, axis=1)
# print(ok.mean())
# print(mv[~ok].head())

# def in_partial(r):
#     st = json.loads(r.state_json)
#     return r.action_id in st["me"]["active"]["moves"]

# print(mv.apply(in_partial, axis=1).mean())

# bad = mv[~ok]
# sp = bad.apply(lambda r: json.loads(r.state_json)["me"]["active"]["species"], axis=1)
# print(Counter(sp).most_common(15))
# print(bad[["battle_id", "turn", "action_id"]].head(10))

# fm = df.drop_duplicates("battle_id")
# z = fm.final_moves_json.str.contains("Zoroark")
# print(z.mean(), z.sum())

# dex = Dex("data/pokedex.json", "data/moves.json")

# # 1. Which Pokédex entries lack types, and what do they contain?
# no_types = {k: v for k, v in dex.pokedex.items() if "types" not in v}
# print(len(no_types))
# for k, v in list(no_types.items())[:5]:
#     print(k, list(v.keys()))

# # 2. Which species in your data hit one of those entries?
# dec = pd.read_parquet("data/decisions_5k.parquet", columns=["state_json"])
# bad = set()
# for s in dec.state_json:
#     st = json.loads(s)
#     for side in ("me", "opp"):
#         for m in [st[side]["active"]] + st[side]["bench"]:
#             if m is None:
#                 continue
#             d = dex.species(m["species"])
#             if d is not None and "types" not in d:
#                 bad.add((m["species"], m["key"]))
# print(sorted(bad)[:20])

# import json, glob

# rows = [json.loads(l) for f in glob.glob("results/decisions_random_*.jsonl")
#         for l in open(f, encoding="utf-8")]
# rows = [r for r in rows if "snapshot" in r and r["turn"] == 1][:3]
# for r in rows:
#     bench = r["snapshot"]["opp"]["bench"]
#     print(r["battle"], [m["species"] or "(padding)" for m in bench])