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
from modules.methods import Preprocessor

JOBLIB_PATH = "joblib/"    

class HorsePastPreprocessor():
    def __init__(self, race_results_df):
        super().__init__()
        self.df_race_results = race_results_df

    def preprocess(self):
        self._remove_space_from_columns()
        self._convert_numeric_values()
        self._split_columns()
        self._process_dates()
        self._process_pass_columns()
        self._drop_unnecessary_columns()
        self._add_time_index_features()
        self._add_season()
        self._get_categorical_values()
        return self.df_race_results
    
    def _remove_space_from_columns(self):
        self.df_race_results.columns = self.df_race_results.columns.str.strip()

    def _convert_numeric_values(self):
        self.df_race_results["着順"] = pd.to_numeric(self.df_race_results["着順"], errors="coerce")
        self.df_race_results.dropna(subset=["着順"], inplace=True)
        self.df_race_results["着順"] = self.df_race_results["着順"].astype(int)
        self.df_race_results["rank"] = self.df_race_results["着順"].map(lambda x: 1 if x < 6 else 0)

        # 性齢を性と年齢に分ける
        self.df_race_results["性"] = self.df_race_results["性齢"].map(lambda x: str(x)[0])
        self.df_race_results["年齢"] = self.df_race_results["性齢"].map(lambda x: str(x)[1:]).astype(int)

        # 馬体重を体重と体重変化に分ける
        self.df_race_results["体重"] = self.df_race_results["馬体重"].str.split("(", expand=True)[0]
        self.df_race_results["体重変化"] = self.df_race_results["馬体重"].str.split("(", expand=True)[1].str[:-1]

        # errors='coerce'で、"計不"など変換できない時に欠損値にする
        self.df_race_results["体重"] = pd.to_numeric(self.df_race_results["体重"], errors="coerce")
        self.df_race_results["体重変化"] = pd.to_numeric(self.df_race_results["体重変化"], errors="coerce")

        # 単勝をfloatに変換
        self.df_race_results["単勝"] = self.df_race_results["単勝"].astype(float)
        # 距離は10の位を切り捨てる
        self.df_race_results["course_len"] = self.df_race_results["course_len"].astype(float) // 100

        def convert_time_to_seconds(time_str):
            # タイムの形式が '分:秒.ミリ秒' と仮定
            # 分と秒を分離
            minutes, seconds = time_str.split(":")
            # 分を秒に変換し、秒と合算
            return int(minutes) * 60 + float(seconds)

        # 'タイム' 列の各エントリに対して変換関数を適用
        self.df_race_results["タイム"] = self.df_race_results["タイム"].apply(convert_time_to_seconds)

    def _split_columns(self):
        self.df_race_results["調教場所"] = self.df_race_results["調教師"].map(lambda x: str(x)[1:2])
        self.df_race_results["調教師名前"] = self.df_race_results["調教師"].map(lambda x: str(x)[3:])

    def _process_dates(self):
        self.df_race_results["date"] = pd.to_datetime(self.df_race_results["date"].str.split("(").str[0])

    def _process_pass_columns(self):
        def split_pass_columns(pass_str):
            # 通過位置を'-'で分割し、最大4つの位置まで取得する
            if "-" not in str(pass_str):
                return [pass_str]
            parts = pass_str.split("-") + [None] * (4 - len(pass_str.split("-")))
            return parts[:4]

        # 新しい列を作成
        self.df_race_results[["通過1", "通過2", "通過3", "通過4"]] = self.df_race_results["通過"].apply(
            lambda x: pd.Series(split_pass_columns(x))
        )

    def _drop_unnecessary_columns(self):
        self.df_race_results.drop(
            ["着差", "通過", "調教師", "性齢", "馬体重", "馬名", "騎手", "人気", "着順", "調教タイム", "厩舎コメント"],
            axis=1,
            inplace=True,
        )

    def _add_time_index_features(self):
        # 日付でソート
        self.df_race_results = self.df_race_results.sort_values(['horse_id', 'date'])

        # 前回のレースからの間隔を計算（日数）
        self.df_race_results['days_since_last_race'] = self.df_race_results.groupby('horse_id')['date'].diff().dt.days

        # 1レース前の情報を追加
        self.df_race_results['prev_タイム指数'] = self.df_race_results.groupby('horse_id')['タイム指数'].shift(1)

        # 集計関数を定義
        def aggregate_time_index(group, n_races):
            return group['タイム指数'].shift().rolling(window=n_races, min_periods=1).agg(['mean', 'max'])

        # 過去3レース、5レースの平均と最大値を計算
        for n in [3, 5]:
            agg_result = self.df_race_results.groupby('horse_id').apply(
                lambda x: aggregate_time_index(x, n)
            ).reset_index()
            self.df_race_results[f'time_index_mean_{n}'] = agg_result['mean']
            self.df_race_results[f'time_index_max_{n}'] = agg_result['max']
            self.df_race_results[f'time_index_min_{n}'] = agg_result['min']

        # 同じ条件下での過去の平均値と最大値を計算
        conditions = ['place', 'course_len', 'weather', 'ground_state']
        for condition in conditions:
            self.df_race_results[f'time_index_mean_{condition}'] = self.df_race_results.groupby(['horse_id', condition])['タイム指数'].transform(
                lambda x: x.shift().expanding().mean()
            )
            self.df_race_results[f'time_index_max_{condition}'] = self.df_race_results.groupby(['horse_id', condition])['タイム指数'].transform(
                lambda x: x.shift().expanding().max()
            )
            self.df_race_results[f'time_index_min_{condition}'] = self.df_race_results.groupby(['horse_id', condition])['タイム指数'].transform(
                lambda x: x.shift().expanding().min()
            )

    def _add_season(self):
        # 季節を追加（春：3-5月、夏：6-8月、秋：9-11月、冬：12-2月）
        self.df_race_results['season'] = pd.Categorical(
            self.df_race_results['date'].dt.month.map({1: '冬', 2: '冬', 3: '春', 4: '春', 5: '春',
                                                        6: '夏', 7: '夏', 8: '夏', 9: '秋', 10: '秋',
                                                        11: '秋', 12: '冬'}),
            categories=['春', '夏', '秋', '冬'], ordered=True
        )

    # カテゴリ変数の処理
    def _get_categorical_values(self):
        if not hasattr(self, 'le_horse'):
            self.le_horse = LabelEncoder()
            self.le_horse.fit(self.df_race_results["horse_id"])

        if not hasattr(self, 'le_jockey'):
            self.le_jockey = LabelEncoder()
            self.le_jockey.fit(self.df_race_results["jockey_id"])

        new_horse_ids = set(self.df_race_results["horse_id"]) - set(self.le_horse.classes_)
        if new_horse_ids:
            self.le_horse.classes_ = np.concatenate([self.le_horse.classes_, list(new_horse_ids)])

        new_jockey_ids = set(self.df_race_results["jockey_id"]) - set(self.le_jockey.classes_)
        if new_jockey_ids:
            self.le_jockey.classes_ = np.concatenate([self.le_jockey.classes_, list(new_jockey_ids)])

        # ラベルエンコーディングを適用
        self.df_race_results["horse_id"] = self.le_horse.transform(self.df_race_results["horse_id"])
        self.df_race_results["jockey_id"] = self.le_jockey.transform(self.df_race_results["jockey_id"])

        # horse_id, jockey_idをpandasのcategory型に変換
        self.df_race_results["horse_id"] = self.df_race_results["horse_id"].astype("category")
        self.df_race_results["jockey_id"] = self.df_race_results["jockey_id"].astype("category")

        # 更新されたLabelEncoderを保存
        joblib.dump(self.le_horse, 'le_horse.joblib')
        joblib.dump(self.le_jockey, 'le_jockey.joblib')

        # そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        # 列を一定にするため
        weathers = ["曇", "晴", "雨", "小雨", "小雪", "雪"]
        race_types = ["芝", "ダート", "障害"]
        ground_states = ["良", "稍重", "重", "不良"]
        sexes = self.df_race_results["性"].unique()
        places = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        self.df_race_results["weather"] = pd.Categorical(self.df_race_results["weather"], weathers)
        self.df_race_results["race_type"] = pd.Categorical(
            self.df_race_results["race_type"], race_types
        )
        self.df_race_results["ground_state"] = pd.Categorical(
            self.df_race_results["ground_state"], ground_states
        )
        self.df_race_results["性"] = pd.Categorical(self.df_race_results["性"], sexes)
        self.df_race_results["place"] = pd.Categorical(self.df_race_results["place"], places)
        self.df_race_results = pd.get_dummies(
            self.df_race_results, columns=["weather", "race_type", "ground_state", "性", "place", "season"]
        )

class PedsPreprocessor():
    def __init__(self, df_peds):
        super().__init__()
        self.df_peds = df_peds

    def get_categorical_values(self):
        for column in self.df_peds.columns:
            self.df_peds[column] = LabelEncoder().fit_transform(self.df_peds[column].fillna("Na"))
        self.df_peds = self.df_peds.astype("category")