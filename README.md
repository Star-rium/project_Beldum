## Requirements:

- Exist a `\data` directory that contains a .parquet file of the dataset (currently only support `formatid: gen9ou`), `moves.json` file that contains all moves data, `pokedex.json` file that contains basic pokemon's data. All of the resources can be downloaded via these links:
[Battle Dataset](https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays/blob/main/%5BGen%209%5D%20OU_part1.parquet?sql_row=2)
[Moves Dataset](https://play.pokemonshowdown.com/data/moves.json)
[Pokedex Dataset](https://play.pokemonshowdown.com/data/pokedex.json)

- Exist a local Pokemon Showdown server in your system
  > Note: not recommended initialize the server in the project directory, put it on somewhere else

```
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
copy config\config-example.js config\config.js
node pokemon-showdown start --no-security

```
## How to use:
(currently working on the client file)
## Todo list:

- [x] Created a decision tree model that used a small subset of the main dataset.
- [x] Initialized the local Pokemon Showdown server and it worked.
- [x] Ran the model against the default Random, Maxpower, Heuristic bots and got results (100% WR on random, 91% on Maxpower, 35% on Heuristic)
- [ ] Adjust the model to encourage rare voluntarily switching
- [ ] Create a one-file client pipeline starting system to effectively run experiments
