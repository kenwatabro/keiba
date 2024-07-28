import sys
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

from modules.Scrape import PedsScraper
from modules.Preprocess import PedsPreprocessor

SAVE_PEDS_DATA_FOLDER = "horse_peds"
HORSE_ID_FILE = "utils/horse_ids.json"


def acquire_peds():
    horse_id_list = _get_horse_id_list()
    scraper = PedsScraper()
    for start in range(0, len(horse_id_list), 1000):
        end = min(start + 1000, len(horse_id_list))
        
        if os.path.exists(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds{start}_{end}.pickle"):
            print(f"horse_peds{start}_{end}.pickle is already exists.")
            continue
        
        df = _get_peds_data(scraper, horse_id_list[start : end])
        _save_peds_files(start, end, df)
        print(f"horse_peds{start}_{end}.pickle is saved.")

def _get_horse_id_list() -> list[str]:
    if os.path.exists(HORSE_ID_FILE):
        with open(HORSE_ID_FILE, "r") as f:
            return list(set(json.load(f)))
    return list(set())


def _get_peds_data(scraper: PedsScraper, horse_id_list: list) -> pd.DataFrame:
    peds = scraper.scrape(horse_id_list)
    return peds


def _save_peds_files(start: int, end: int, df_peds: pd.DataFrame):
    df_peds.to_pickle(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds{start}_{end}.pickle")

def main():
    acquire_peds()

if __name__ == "__main__":
    main()