import sys
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

from modules.Scrape import PedsScraper
from modules.Preprocess import PedsPreprocessor

SAVE_PEDS_DATA_FOLDER = "horse_peds"
HORSE_ID_FILE = "horse_ids.json"


def acqure_peds():
    horse_id_list = _get_horse_id_list()
    for start in range(0, len(horse_id_list), 1000):
        if os.path.exists(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds{start}.pickle"):
            print(f"horse_peds{start}.pickle is already exists.")
            break
        df = _get_peds_data(horse_id_list[start : start + 1000])
        _save_peds_files(start, df)
        print(f"horse_peds{start}.pickle is saved.")

def _get_horse_id_list() -> list[str]:
    if os.path.exists(HORSE_ID_FILE):
        with open(HORSE_ID_FILE, "r") as f:
            return list(set(json.load(f)))
    return list(set())


def _get_peds_data(horse_id_list: list) -> pd.DataFrame:
    peds_df = pd.DataFrame()
    for horse_id in horse_id_list:
        peds_df = pd.concat([peds_df, pd.read_pickle(f"peds/{horse_id}.pickle")])
    return peds_df


def _save_peds_files(start: int, df_peds: pd.DataFrame):
    df_peds.to_pickle(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds{start}.pickle")
