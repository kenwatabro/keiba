import os
import pandas as pd
import numpy as np
import datetime
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

from modules.Preprocess import HorsePastPreprocessor

DATA_PATH = "horse_preprocessed/Race_preprocessed.pickle"

def get_data():
    if os.path.exists(DATA_PATH):
        data = pd.read_pickle(DATA_PATH)
    else:
        data = _preprocess_horse_data()
    return data

def _preprocess_horse_data():
    df_race_result_all = pd.DataFrame()
    for file in os.listdir("horse_results"):
        df_race_results = pd.read_pickle(f"horse_results/{file}")
        if df_race_results.empty:
            continue
        df_race_result_all = pd.concat([df_race_result_all, df_race_results])

    hpp = HorsePastPreprocessor(df_race_result_all)
    df_race_result_all_preprocessed = hpp.preprocess()

    df_race_result_all_preprocessed.to_pickle(
        "horse_preprocessed/Race_preprocessed.pickle"
    )
    return df_race_result_all_preprocessed

def split_data(df, test_size=0.3):
    sorted_id_list = df.sort_values("date").index.unique()
    train_id_list = sorted_id_list[: round(len(sorted_id_list) * (1 - test_size))]
    test_id_list = sorted_id_list[round(len(sorted_id_list) * (1 - test_size)) :]
    train = df.loc[train_id_list]
    test = df.loc[test_id_list]
    return train, test

if __name__ == "__main__":
    data = get_data()
    data = data.drop(columns=["タイム指数"])
    train, test = split_data(data)
    train, valid = split_data(train)
    
    X_train = train.drop(["rank", "date"])
    y_train = train["rank"]
    X_valid = valid.drop(["rank", "date"])
    y_valid = valid["rank"]
    X_test = test.drop(["rank", "date"])
    y_test = test["rank"]
    
    lgb_train = lgb_o.Dataset(X_train.values, y_train.values)
    lgb_valid = lgb_o.Dataset(X_valid.values, y_valid.values)
    lgb_test = lgb_o.Dataset(X_test.values, y_test.values)
    
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "seed": 42,
    }
    
    lgb_clf_o = lgb_o.train(params, lgb_train, valid_sets=(lgb_train, lgb_valid), early_stopping_rounds=10, verbose_eval=100, optuna_seed=100)
    
    