import sys
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

from modules.Scrape import RaceScraper
from modules.Preprocess import HorsePastPreprocessor

SAVE_RACE_DATA_FOLDER = "horse_results"
HORSE_ID_FILE = "utils/horse_ids.json"


def main():
    trace_back_year = _get_arguments()
    acquire_race(trace_back_year)

def _get_arguments():
    # コマンドライン引数を取得
    args = sys.argv[1:]  # 最初の要素(スクリプト名)を除く
    return int(args[0])

def acquire_race(trace_back_year):
    scraper = RaceScraper()
    year, month, day = datetime.now().year, datetime.now().month, datetime.now().day
    start_year = f"{year - trace_back_year}"
    end_year = year

    for year in range(int(start_year), int(end_year) + 1, 1):
        race_id_no_round_list = _get_race_id_no_round_list(year)

        for start in range(0, len(race_id_no_round_list), 100):
            end = min(start + 100, len(race_id_no_round_list))
            print(f"acquiring race data: year={year}, start={start}, end={end}")

            if os.path.exists(f"{SAVE_RACE_DATA_FOLDER}/Race_{year}_{start}-{end}.pickle"):
                print(f"Race_{year}_{start}-{end}.pickle is already exists.")
                continue
            
            df = _get_race_data(scraper, year, start, race_id_no_round_list[start:end])
            _save_files(year, start, df)


def _get_race_id_no_round_list(year: str) -> list[str]:  # 引数で指定された年のレースID一覧:
    race_id_list = []
    for place in range(1, 11, 1):
        for kai in range(1, 7, 1):
            for day in range(1, 13, 1):
                # for r in range(1, 13, 1):
                race_id = (
                    str(year)
                    + str(place).zfill(2)
                    + str(kai).zfill(2)
                    + str(day).zfill(2)
                    # + str(r).zfill(2)
                )
                race_id_list.append(race_id)
    return race_id_list


def _get_race_data(scraper: RaceScraper, year: str, start: int, race_id_list: list[str]) -> pd.DataFrame:
    race_res = scraper.scrape(race_id_list)

    # horse_idのリストを更新
    existing_horse_ids = _load_horse_ids()
    new_horse_ids = set(race_res["horse_id"].unique())
    updated_horse_ids = existing_horse_ids.union(new_horse_ids)
    _save_horse_ids(updated_horse_ids)

    return race_res


def _load_horse_ids():
    if os.path.exists(HORSE_ID_FILE):
        with open(HORSE_ID_FILE, "r") as f:
            return set(json.load(f))
    return set()


def _save_horse_ids(horse_ids):
    with open(HORSE_ID_FILE, "w") as f:
        json.dump(list(horse_ids), f)


def _save_files(year: str, start: int, df_horse_results: pd.DataFrame):
    df_horse_results.to_pickle(
        f"{SAVE_RACE_DATA_FOLDER}/Race_{year}_{start}-{start+100}.pickle"
    )
    print(f"Race_{year}_{start}-{start+100}.pickle is saved.")

if __name__ == "__main__":
    main()
