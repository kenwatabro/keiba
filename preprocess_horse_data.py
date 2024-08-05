import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib
import os

import login_info
from modules.Preprocess import HorsePastPreprocessor, HorseTodayPreprocessor
from modules.Scrape import TodaysRaceScraper


df_race_result_2020_2024 = pd.DataFrame()
for year in range(2020, 2025):
    for place in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]:
        df_race_results = pd.read_pickle(f"horse_results/Race_{year}_{place}.pickle")
        if df_race_results.empty:
            continue
        df_race_result_2020_2024 = pd.concat(
            [df_race_result_2020_2024, df_race_results]
        )

hpp = HorsePastPreprocessor(df_race_result_2020_2024)
df_race_result_2020_2024_preprocessed = hpp.preprocess()



df_race_result_2020_2024_preprocessed.to_pickle("horse_results/Race_preprocessed.pickle")


# todays_race_scraper = TodaysRaceScraper()
# todays_race_scraper.get_todays_race_id_n_time()
# race_id_list = list(todays_race_scraper.todays_race_time.keys())
# race = todays_race_scraper.scrape("202404020407")

# htp = HorseTodayPreprocessor(race, df_race_result_2020_2024_preprocessed)
# race_preprocessed = htp.preprocess()