import sys
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

from modules.Scrape import PedsScraper
from modules.Preprocess import PedsPreprocessor

SAVE_PEDS_DATA_FOLDER = "horse_peds"


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
    df_horse_result = pd.DataFrame()
    for year in range(2020, 2025):
        for place in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
            df = pd.read_pickle(f"horse_results/Race_{year}_{place}.pickle")
        if df.empty:
            continue
        df_horse_result = pd.concat([df_horse_result, df])
    return list(df_horse_result.horse_id.unique())


def _get_peds_data(scraper: PedsScraper, horse_id_list: list) -> pd.DataFrame:
    peds = scraper.scrape(horse_id_list)
    return peds


def _save_peds_files(start: int, end: int, df_peds: pd.DataFrame):
    df_peds.to_pickle(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds{start}_{end}.pickle")

def main():
    acquire_peds()

if __name__ == "__main__":
    main()
