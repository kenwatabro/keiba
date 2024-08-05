import pandas as pd
import numpy as np
import datetime
import os
import pickle
from tqdm.notebook import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import time
import re
from urllib.request import urlopen
import optuna.integration.lightgbm as lgb_o
from itertools import combinations, permutations
import matplotlib.pyplot as plt

from modules.Scrape import RaceScraper, TodaysRaceScraper
from modules.Preprocess import HorseTodayPreprocessor

MOEDL_FOLDER_PATH = "trained_model"


data = pd.read_pickle("horse_preprocessed/Race_preprocessed.pickle")
def predict_proba(new_data):
    # モデルを読み込む
    with open("trained_models/lgb_model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    predictions = loaded_model.predict(new_data.values)

    return predictions

trs = TodaysRaceScraper()
race = trs.scrape("202404020408")
htp = HorseTodayPreprocessor(race, data)
race_preprocessed = htp.preprocess()

race_preprocessed = race_preprocessed.drop(columns=["date"])
predict_proba(race_preprocessed)
