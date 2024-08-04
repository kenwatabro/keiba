import sys
import numpy as np
import pandas as pd
from datetime import datetime
import os

from modules.Scrape import RaceScraper


SAVE_RACE_DATA_FOLDER = "horse_results"
SAVE_ODDS_DATA_FOLDER = "odds_results"
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

        for start in range(0, len(race_id_no_round_list), 72):
            end = min(start + 72, len(race_id_no_round_list))
            place = race_id_no_round_list[start][4:6]
            print(f"acquiring race data: year={year}, start={start}, end={end}")

            if os.path.exists(f"{SAVE_RACE_DATA_FOLDER}/Race_{year}_{place}.pickle"):
                print(f"Race_{year}_{place}.pickle is already exists.")
                continue

            df, odds_df = scraper.scrape(race_id_no_round_list[start:end])

            _save_files(year, place, df, odds_df)


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
                )
                race_id_list.append(race_id)
    return race_id_list




def _save_files(year: str, place: int, df_horse_results: pd.DataFrame, df_odds: pd.DataFrame):
    df_horse_results["place"] = str(place)
    df_horse_results.to_pickle(
        f"{SAVE_RACE_DATA_FOLDER}/Race_{year}_{place}.pickle"
    )
    df_odds.to_pickle(
        f"{SAVE_ODDS_DATA_FOLDER}/Odds_{year}_{place}.pickle"
    )
    print(f"Race_{year}_{place}.pickle is saved.")

if __name__ == "__main__":
    main()
