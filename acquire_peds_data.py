import numpy as np
import pandas as pd
from datetime import datetime
import os

from modules.Scrape import PedsScraper
from modules.Preprocess import PedsPreprocessor

SAVE_PEDS_DATA_FOLDER = "horse_peds"


def acquire_peds():
    if os.path.exists(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds_all.pickle"):
        print("updating peds data...")
        horse_id_set_peds = set(pd.read_pickle(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds_all.pickle").index)
        additional = True
    elif os.path.exists(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds0_1000.pickle"):
        print("merging and updating peds data...")
        df_peds_all = pd.DataFrame()
        for file in os.listdir(SAVE_PEDS_DATA_FOLDER):
            if file.endswith(".pickle"):
                df_peds = pd.read_pickle(f"{SAVE_PEDS_DATA_FOLDER}/{file}")
                df_peds_all = pd.concat([df_peds_all, df_peds])
        df_peds_all.to_pickle(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds_all.pickle")
        horse_id_set_peds = set(df_peds_all.index)
        additional = True
    else:
        print("No peds data found.")
        horse_id_set_peds = set()
        additional = False
    horse_id_set_race = _get_horse_id_list()
    horse_id_list = list(horse_id_set_race - horse_id_set_peds)
    scraper = PedsScraper()

    for start in range(0, len(horse_id_list), 1000):
        end = min(start + 1000, len(horse_id_list))
        df_peds = _get_peds_data(scraper, horse_id_list[start : end])
        _save_peds_files(start, end, df_peds, df_peds_all, additional)


def _get_horse_id_list() -> list[str]:
    df_horse_result = pd.DataFrame()
    for year in range(2020, 2025):
        for place in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
            df = pd.read_pickle(f"horse_results/Race_{year}_{place}.pickle")
            if df.empty:
                continue
            df_horse_result = pd.concat([df_horse_result, df])
    return set(df_horse_result.horse_id.unique())


def _get_peds_data(scraper: PedsScraper, horse_id_list: list) -> pd.DataFrame:
    peds = scraper.scrape(horse_id_list)
    return peds


def _save_peds_files(start: int, end: int, df_peds: pd.DataFrame, df_peds_all: pd.DataFrame, additional: bool):
    if additional:
        df_peds_all = pd.concat([df_peds_all, df_peds])
        df_peds_all.to_pickle(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds_all.pickle")
        print("added to horse_peds_all.pickle.")
    else:
        df_peds.to_pickle(f"{SAVE_PEDS_DATA_FOLDER}/horse_peds{start}_{end}.pickle")
        print(f"horse_peds{start}_{end}.pickle is saved.")


if __name__ == "__main__":
    acquire_peds()
