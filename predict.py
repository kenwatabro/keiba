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
def predict_proba(new_data, model_path="trained_models/lgb_model.pkl"):
    # モデルを読み込む
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    # 予測確率を計算
    predictions = loaded_model.predict_proba(new_data.values)[:, 1]
    
    # 予測結果をデータフレームに追加
    new_data["予測確率"] = predictions
    
    # 投資判断（閾値は調整可能）
    threshold = 0.1
    new_data["投資判断"] = new_data["予測確率"] > threshold
    
    return new_data

def evaluate_predictions(predictions, actual_results):
    # 投資シミュレーション
    bets = predictions[predictions["投資判断"]]
    total_bets = len(bets)
    winning_bets = len(bets[bets["単勝的中"] == 1])
    total_return = (bets[bets["単勝的中"] == 1]["単勝オッズ"]).sum()
    
    print(f"総投資回数: {total_bets}")
    print(f"的中回数: {winning_bets}")
    print(f"回収率: {(total_return/total_bets)*100:.2f}%")
    print(f"ROI: {(total_return/total_bets):.2f}")
    
    return {
        "total_bets": total_bets,
        "winning_bets": winning_bets,
        "return_rate": (total_return/total_bets)*100,
        "roi": total_return/total_bets
    }

trs = TodaysRaceScraper()
race = trs.scrape("202404020408")
htp = HorseTodayPreprocessor(race, data)
race_preprocessed = htp.preprocess()

race_preprocessed = race_preprocessed.drop(columns=["date"])
predict_proba(race_preprocessed)
