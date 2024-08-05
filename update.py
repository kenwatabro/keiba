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

def update(old_data: pd.DataFrame, new_data: pd.DataFrame):
    filtered_old = old_data[~old_data.index.isin(new_data.index)]
    filtered_new = new_data[~new_data.index.isin(old_data.index)]
    return pd.concat([filtered_old, filtered_new], ignore_index=True)

