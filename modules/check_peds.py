import os
import pandas as pd

horse_id_set_peds = set()
for file in os.listdir("horse_peds"):
    horse_ids = pd.read_pickle(f"horse_peds/{file}").index.unique()
    horse_id_set_peds.update(horse_ids)

horse_id_set_race = set()
for file in os.listdir("horse_results"):
    if file.endswith(".pickle"):
        if pd.read_pickle(f"horse_results/{file}").empty:
            continue
        horse_ids = pd.read_pickle(f"horse_results/{file}").horse_id.unique()
        horse_id_set_race.update(horse_ids)


only_in_race = horse_id_set_race - horse_id_set_peds


