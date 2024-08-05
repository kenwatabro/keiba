import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from modules.methods import Preprocessor

JOBLIB_PATH = "joblib/"


class HorsePastPreprocessor(Preprocessor):
    def __init__(self, race_results_df):
        super().__init__()
        self.df_race_results = race_results_df.copy()

    def preprocess(self):
        self._remove_space_from_columns()
        self._convert_numeric_values()
        self._split_columns()
        self._process_dates()
        self._process_pass_columns()
        self._process_agari()
        self._add_season()
        self._get_categorical_values()
        self._add_time_index_features()
        self._drop_unnecessary_columns()
        return self.df_race_results

    def _remove_space_from_columns(self):
        self.df_race_results.columns = self.df_race_results.columns.str.strip()

    def _convert_numeric_values(self):
        self.df_race_results["着順"] = pd.to_numeric(
            self.df_race_results["着順"], errors="coerce"
        )
        self.df_race_results.dropna(subset=["着順"], inplace=True)
        self.df_race_results["着順"] = self.df_race_results["着順"].astype(int)
        self.df_race_results["rank"] = self.df_race_results["着順"].map(
            lambda x: 1 if x < 6 else 0
        )

        # 性齢を性と年齢に分ける
        self.df_race_results["性"] = self.df_race_results["性齢"].map(
            lambda x: str(x)[0]
        )
        self.df_race_results["年齢"] = (
            self.df_race_results["性齢"].map(lambda x: str(x)[1:]).astype(int)
        )

        # 馬体重を体重と体重変化に分ける
        self.df_race_results["体重"] = self.df_race_results["馬体重"].str.split(
            "(", expand=True
        )[0]
        self.df_race_results["体重変化"] = (
            self.df_race_results["馬体重"].str.split("(", expand=True)[1].str[:-1]
        )

        # errors='coerce'で、"計不"など変換できない時に欠損値にする
        self.df_race_results["体重"] = pd.to_numeric(
            self.df_race_results["体重"], errors="coerce"
        )
        self.df_race_results["体重変化"] = pd.to_numeric(
            self.df_race_results["体重変化"], errors="coerce"
        )

        # 単勝をfloatに変換
        self.df_race_results["単勝"] = self.df_race_results["単勝"].astype(float)
        # 距離は10の位を切り捨てる
        self.df_race_results["course_len"] = (
            self.df_race_results["course_len"].astype(float) // 100
        )

        def convert_time_to_seconds(time_str):
            # タイムの形式が '分:秒.ミリ秒' と仮定
            # 分と秒を分離
            minutes, seconds = time_str.split(":")
            # 分を秒に変換し、秒と合算
            return int(minutes) * 60 + float(seconds)

        # 'タイム' 列の各エントリに対して変換関数を適用
        self.df_race_results["タイム"] = self.df_race_results["タイム"].apply(
            convert_time_to_seconds
        )

    def _split_columns(self):
        self.df_race_results["調教場所"] = self.df_race_results["調教師"].map(
            lambda x: str(x)[1:2]
        )
        self.df_race_results["調教師名前"] = self.df_race_results["調教師"].map(
            lambda x: str(x)[3:]
        )

    def _process_dates(self):
        # ['勝馬投票100周年' '三年坂ステークス' '競馬法100周年記念' '府中市70周年記念' '宝塚市制70周年記念']は個別処理
        self.df_race_results.loc[
            self.df_race_results["date"] == "勝馬投票100周年", "date"
        ] = pd.to_datetime("2023/08/27")
        self.df_race_results.loc[
            self.df_race_results["date"] == "三年坂ステークス", "date"
        ] = pd.to_datetime("2023/10/22")
        self.df_race_results.loc[
            self.df_race_results["date"] == "競馬法100周年記念", "date"
        ] = pd.to_datetime("2023/04/09")
        self.df_race_results.loc[
            self.df_race_results["date"] == "府中市70周年記念", "date"
        ] = pd.to_datetime("2024/04/27")
        self.df_race_results.loc[
            self.df_race_results["date"] == "宝塚市制70周年記念", "date"
        ] = pd.to_datetime("2024/04/07")
        self.df_race_results["date"] = pd.to_datetime(
            self.df_race_results["date"].str.split("(").str[0]
        )

    def _process_pass_columns(self):
        def split_pass_columns(pass_str):
            # 通過位置を'-'で分割し、最大4つの位置まで取得する
            if "-" not in str(pass_str):
                return [pass_str]
            parts = pass_str.split("-") + [None] * (4 - len(pass_str.split("-")))
            return parts[:4]

        # 新しい列を作成
        self.df_race_results[["通過1", "通過2", "通過3", "通過4"]] = (
            self.df_race_results["通過"].apply(
                lambda x: pd.Series(split_pass_columns(x))
            )
        )

        self.df_race_results["頭数"] = self.df_race_results.groupby(level=0)[
            "馬番"
        ].transform("max")

        def calculate_run_type(row):
            total_horses = row["頭数"]
            pass_position = row["通過1"]

            if pd.isna(pass_position):
                return None

            relative_position = int(pass_position) / total_horses

            return relative_position
            # 逃げ先行差し追い込み
            # if relative_position <= 0.15:
            #     return 1
            # elif relative_position <= 0.50:
            #     return 2
            # elif relative_position <= 0.85:
            #     return 3
            # else:
            #     return 4

        self.df_race_results["run_type"] = self.df_race_results.apply(
            calculate_run_type, axis=1
        )

    def _process_agari(self):
        # Convert '上り' to numeric, replacing '-' with NaN
        self.df_race_results["上り"] = pd.to_numeric(
            self.df_race_results["上り"].replace("-", np.nan), errors="coerce"
        )

        # Sort by horse_id and date
        self.df_race_results = self.df_race_results.sort_values(["horse_id", "date"])

        # Calculate the rolling mean of '上り' for each horse, excluding the current race
        self.df_race_results["avg_上り"] = self.df_race_results.groupby("horse_id")[
            "上り"
        ].transform(lambda x: x.shift().expanding().mean())

    def _add_season(self):
        # 季節を追加（春：3-5月、夏：6-8月、秋：9-11月、冬：12-2月）
        self.df_race_results["season"] = pd.Categorical(
            self.df_race_results["date"].dt.month.map(
                {
                    1: "冬",
                    2: "冬",
                    3: "春",
                    4: "春",
                    5: "春",
                    6: "夏",
                    7: "夏",
                    8: "夏",
                    9: "秋",
                    10: "秋",
                    11: "秋",
                    12: "冬",
                }
            ),
            categories=["春", "夏", "秋", "冬"],
            ordered=True,
        )

    # カテゴリ変数の処理
    def _get_categorical_values(self):
        # Load existing LabelEncoders if they exist
        if os.path.exists("le_horse.joblib"):
            self.le_horse = joblib.load("le_horse.joblib")
        else:
            self.le_horse = LabelEncoder()
            self.le_horse.fit(self.df_race_results["horse_id"])

        if os.path.exists("le_jockey.joblib"):
            self.le_jockey = joblib.load("le_jockey.joblib")
        else:
            self.le_jockey = LabelEncoder()
            self.le_jockey.fit(self.df_race_results["jockey_id"])

        if os.path.exists("le_chokyoshi.joblib"):
            self.le_chokyoshi = joblib.load("le_chokyoshi.joblib")
        else:
            self.le_chokyoshi = LabelEncoder()
            self.le_chokyoshi.fit(self.df_race_results["調教師名前"])

        # Update classes if new categories are found
        new_horse_ids = set(self.df_race_results["horse_id"]) - set(
            self.le_horse.classes_
        )
        if new_horse_ids:
            self.le_horse.classes_ = np.concatenate(
                [self.le_horse.classes_, list(new_horse_ids)]
            )

        new_jockey_ids = set(self.df_race_results["jockey_id"]) - set(
            self.le_jockey.classes_
        )
        if new_jockey_ids:
            self.le_jockey.classes_ = np.concatenate(
                [self.le_jockey.classes_, list(new_jockey_ids)]
            )

        new_chokyoshi_ids = set(self.df_race_results["調教師名前"]) - set(
            self.le_chokyoshi.classes_
        )
        if new_chokyoshi_ids:
            self.le_chokyoshi.classes_ = np.concatenate(
                [self.le_chokyoshi.classes_, list(new_chokyoshi_ids)]
            )

        # Apply label encoding
        self.df_race_results["horse_id"] = self.le_horse.transform(
            self.df_race_results["horse_id"]
        )
        self.df_race_results["jockey_id"] = self.le_jockey.transform(
            self.df_race_results["jockey_id"]
        )
        self.df_race_results["調教師名前"] = self.le_chokyoshi.transform(
            self.df_race_results["調教師名前"]
        )

        # Convert to category type
        self.df_race_results["horse_id"] = self.df_race_results["horse_id"].astype(
            "category"
        )
        self.df_race_results["jockey_id"] = self.df_race_results["jockey_id"].astype(
            "category"
        )
        self.df_race_results["調教師名前"] = self.df_race_results["調教師名前"].astype(
            "category"
        )

        # Save updated LabelEncoders
        joblib.dump(self.le_horse, "le_horse.joblib")
        joblib.dump(self.le_jockey, "le_jockey.joblib")
        joblib.dump(self.le_chokyoshi, "le_chokyoshi.joblib")

        # そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        # 列を一定にするため
        weathers = ["曇", "晴", "雨", "小雨", "小雪", "雪"]
        race_types = ["芝", "ダート", "障害"]
        ground_states = ["良", "稍", "重", "不"]
        sexes = self.df_race_results["性"].unique()
        places = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

        self.df_race_results["weather"] = pd.Categorical(
            self.df_race_results["weather"], weathers
        )
        self.df_race_results["race_type"] = pd.Categorical(
            self.df_race_results["race_type"], race_types
        )
        self.df_race_results["ground_state"] = pd.Categorical(
            self.df_race_results["ground_state"], ground_states
        )
        self.df_race_results["性"] = pd.Categorical(self.df_race_results["性"], sexes)
        self.df_race_results["place"] = pd.Categorical(
            self.df_race_results["place"], places
        )

        self.df_race_results = pd.get_dummies(
            self.df_race_results,
            columns=[
                "weather",
                "race_type",
                "ground_state",
                "性",
                "place",
                "season",
            ],
        )

        self.df_race_results["調教場所"] = (
            self.df_race_results["調教場所"] == "東"
        ).astype("category")

    def _add_time_index_features(self):
        # 日付でソート
        self.df_race_results = self.df_race_results.sort_values(["horse_id", "date"])

        # 前回のレースからの間隔を計算（日数）
        self.df_race_results["days_since_last_race"] = (
            self.df_race_results.groupby("horse_id")["date"].diff().dt.days
        )

        # 1レース前��報を追加
        self.df_race_results["prev_タイム指数"] = self.df_race_results.groupby(
            "horse_id"
        )["タイム指数"].shift(1)

        # 集計関数を定義
        def aggregate_time_index(group, n_races):
            return (
                group["タイム指数"]
                .shift()
                .rolling(window=n_races, min_periods=1)
                .agg(["mean", "max", "min"])
            )

        # 過去3レース、5レースの平均、最大値、最小値を計算
        for n in [3, 5]:
            agg_result = (
                self.df_race_results.groupby("horse_id")
                .apply(lambda x: aggregate_time_index(x, n))
                .reset_index()
            )
            self.df_race_results[f"time_index_mean_{n}"] = agg_result["mean"]
            self.df_race_results[f"time_index_max_{n}"] = agg_result["max"]
            self.df_race_results[f"time_index_min_{n}"] = agg_result["min"]
        # 同じ条件下での過去の平均値と最大値を計算
        condition_columns = {
            "weather": [
                "weather_曇",
                "weather_晴",
                "weather_雨",
                "weather_小雨",
                "weather_小雪",
                "weather_雪",
            ],
            "race_type": ["race_type_芝", "race_type_ダート", "race_type_障害"],
            "ground_state": [
                "ground_state_良",
                "ground_state_稍",
                "ground_state_重",
                "ground_state_不",
            ],
            "place": [
                "place_01",
                "place_02",
                "place_03",
                "place_04",
                "place_05",
                "place_06",
                "place_07",
                "place_08",
                "place_09",
                "place_10",
            ],
        }

        for condition, columns in condition_columns.items():
            for column in columns:
                agg_result = (
                    self.df_race_results.sort_values(["horse_id", "date"])
                    .groupby(["horse_id"])
                    .apply(
                        lambda x: x[x[column]]["prev_タイム指数"].agg(
                            ["mean", "max", "min"]
                        )
                    )
                    .reset_index()
                )
                self.df_race_results = self.df_race_results.merge(
                    agg_result, on="horse_id", how="left", suffixes=("", f"_{column}")
                )
                self.df_race_results = self.df_race_results.rename(
                    columns={
                        "mean": f"time_index_mean_{column}",
                        "max": f"time_index_max_{column}",
                        "min": f"time_index_min_{column}",
                    }
                )

        # course_lenは数値なので別途処理
        agg_result = (
            self.df_race_results.sort_values(["horse_id", "date"])
            .groupby(["horse_id", "course_len"])["prev_タイム指数"]
            .agg(["mean", "max", "min"])
            .reset_index()
        )
        self.df_race_results = self.df_race_results.merge(
            agg_result,
            on=["horse_id", "course_len"],
            how="left",
            suffixes=("", "_course_len"),
        )
        self.df_race_results = self.df_race_results.rename(
            columns={
                "mean": "time_index_mean_course_len",
                "max": "time_index_max_course_len",
                "min": "time_index_min_course_len",
            }
        )

    def _drop_unnecessary_columns(self):
        self.df_race_results.drop(
            [
                "単勝",
                "着差",
                "通過",
                "調教師",
                "馬主",
                "性齢",
                "タイム",
                "ground_index",
                "備考",
                "馬体重",
                "賞金（万円）",
                "馬名",
                "騎手",
                "人気",
                "着順",
                "調教タイム",
                "厩舎コメント",
                "通過1",
                "通過2",
                "通過3",
                "通過4",
            ],
            axis=1,
            inplace=True,
        )


class PedsPreprocessor(Preprocessor):
    def __init__(self, df_peds):
        super().__init__()
        self.df_peds = df_peds.copy()

    def preprocess(self):
        for column in self.df_peds.columns:
            self.df_peds[column] = LabelEncoder().fit_transform(
                self.df_peds[column].fillna("Na")
            )
        self.df_peds = self.df_peds.astype("category")


class HorseTodayPreprocessor(Preprocessor):
    def __init__(self, df_shutsuba_table, df_horse_past):
        super().__init__()
        self.df_shutsuba_table = df_shutsuba_table.copy()
        self.df_horse_past = df_horse_past.copy()
        self.df_target_horses: pd.DataFrame

    def preprocess(self):
        self._convert_ids_to_category()
        self._get_target_horses_info()
        self._remove_space_from_columns()
        self._convert_numeric_values()
        self._process_dates()
        self._add_tosuu()
        self._get_estimated_runtype()
        self._get_estimated_agari()
        self._add_season()
        self._add_chokyo_info()
        self._get_categorical_values()
        self._add_time_index_features()  # categorical に変換後でないとdf_horse_pastと不整合
        self._drop_and_rename_columns()
        return self.df_shutsuba_table

    def _convert_ids_to_category(self):
        # Load existing LabelEncoders if they exist
        if os.path.exists("le_horse.joblib"):
            self.le_horse = joblib.load("le_horse.joblib")
        else:
            self.le_horse = LabelEncoder()
            self.le_horse.fit(self.df_shutsuba_table["horse_id"])

        if os.path.exists("le_jockey.joblib"):
            self.le_jockey = joblib.load("le_jockey.joblib")
        else:
            self.le_jockey = LabelEncoder()
            self.le_jockey.fit(self.df_shutsuba_table["jockey_id"])

        # Update classes if new categories are found
        new_horse_ids = set(self.df_shutsuba_table["horse_id"]) - set(
            self.le_horse.classes_
        )
        if new_horse_ids:
            self.le_horse.classes_ = np.concatenate(
                [self.le_horse.classes_, list(new_horse_ids)]
            )

        new_jockey_ids = set(self.df_shutsuba_table["jockey_id"]) - set(
            self.le_jockey.classes_
        )
        if new_jockey_ids:
            self.le_jockey.classes_ = np.concatenate(
                [self.le_jockey.classes_, list(new_jockey_ids)]
            )

        # Apply label encoding
        self.df_shutsuba_table["horse_id"] = self.le_horse.transform(
            self.df_shutsuba_table["horse_id"]
        )
        self.df_shutsuba_table["jockey_id"] = self.le_jockey.transform(
            self.df_shutsuba_table["jockey_id"]
        )

        # Convert to category type
        self.df_shutsuba_table["horse_id"] = self.df_shutsuba_table["horse_id"].astype(
            "category"
        )
        self.df_shutsuba_table["jockey_id"] = self.df_shutsuba_table[
            "jockey_id"
        ].astype("category")

        # Save updated LabelEncoders
        joblib.dump(self.le_horse, "le_horse.joblib")
        joblib.dump(self.le_jockey, "le_jockey.joblib")

    def _get_target_horses_info(self):
        self.df_target_horses = self.df_horse_past[
            self.df_horse_past["horse_id"].isin(
                self.df_shutsuba_table["horse_id"].unique()
            )
        ]

    def _remove_space_from_columns(self):
        self.df_shutsuba_table.columns = self.df_shutsuba_table.columns.str.strip()

    def _convert_numeric_values(self):

        # 性齢を性と年齢に分ける
        self.df_shutsuba_table["性"] = self.df_shutsuba_table["性齢"].map(
            lambda x: str(x)[0]
        )
        self.df_shutsuba_table["年齢"] = (
            self.df_shutsuba_table["性齢"].map(lambda x: str(x)[1:]).astype(int)
        )

        # 馬体重を体重と体重変化に分ける
        self.df_shutsuba_table["体重"] = self.df_shutsuba_table[
            "馬体重(増減)"
        ].str.split("(", expand=True)[0]
        self.df_shutsuba_table["体重変化"] = (
            self.df_shutsuba_table["馬体重(増減)"]
            .str.split("(", expand=True)[1]
            .str[:-1]
        )

        # errors='coerce'、"計不"など変換できない時に欠損値にする
        self.df_shutsuba_table["体重"] = pd.to_numeric(
            self.df_shutsuba_table["体重"], errors="coerce"
        )
        self.df_shutsuba_table["体重変化"] = pd.to_numeric(
            self.df_shutsuba_table["体重変化"], errors="coerce"
        )

        # 距離は10の位を切り捨てる
        self.df_shutsuba_table["course_len"] = (
            self.df_shutsuba_table["course_len"].astype(float) // 100
        )

    def _process_dates(self):
        self.df_shutsuba_table["date"] = pd.Timestamp("now")

    def _add_tosuu(self):
        self.df_shutsuba_table["頭数"] = len(self.df_shutsuba_table)

    def _get_estimated_runtype(self):

        # Calculate average run_type for each relevant horse
        avg_run_type = self.df_target_horses.groupby("horse_id")["run_type"].mean()

        # Merge with df_shutsuba_table
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            avg_run_type.rename("run_type"), on="horse_id", how="left"
        )

    def _get_estimated_agari(self):
        avg_run_type = self.df_target_horses.groupby("horse_id")["上り"].mean()

        # Merge with df_shutsuba_table
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            avg_run_type.rename("avg_上り"), on="horse_id", how="left"
        )

    def _add_time_index_features(self):

        # 集計関数を定義
        def aggregate_time_index(group, n_races):
            return group["タイム指数"].tail(n_races).agg(["mean", "max", "min"])

        # 過去3レース、5レースの平均、最大値、最小値を計算
        for n in [3, 5]:
            agg_result = (
                self.df_target_horses.sort_values(["horse_id", "date"])
                .groupby("horse_id")
                .apply(lambda x: aggregate_time_index(x, n))
                .reset_index()
            )
            self.df_shutsuba_table = self.df_shutsuba_table.merge(
                agg_result, on="horse_id", how="left", suffixes=("", f"_last_{n}")
            )
            self.df_shutsuba_table = self.df_shutsuba_table.rename(
                columns={
                    "mean": f"time_index_mean_{n}",
                    "max": f"time_index_max_{n}",
                    "min": f"time_index_min_{n}",
                }
            )

        # 同じ条件下での過去の平均値、最大値、最小値を計算
        condition_columns = {
            "weather": [
                "weather_曇",
                "weather_晴",
                "weather_雨",
                "weather_小雨",
                "weather_小雪",
                "weather_雪",
            ],
            "race_type": ["race_type_芝", "race_type_ダート", "race_type_障害"],
            "ground_state": [
                "ground_state_良",
                "ground_state_稍",
                "ground_state_重",
                "ground_state_不",
            ],
            "place": [
                "place_01",
                "place_02",
                "place_03",
                "place_04",
                "place_05",
                "place_06",
                "place_07",
                "place_08",
                "place_09",
                "place_10",
            ],
        }

        for condition, columns in condition_columns.items():
            for column in columns:
                agg_result = (
                    self.df_target_horses.sort_values(["horse_id", "date"])
                    .groupby(["horse_id"])
                    .apply(
                        lambda x: x[x[column]]["タイム指数"].agg(["mean", "max", "min"])
                    )
                    .reset_index()
                )
                self.df_shutsuba_table = self.df_shutsuba_table.merge(
                    agg_result, on="horse_id", how="left", suffixes=("", f"_{column}")
                )
                self.df_shutsuba_table = self.df_shutsuba_table.rename(
                    columns={
                        "mean": f"time_index_mean_{column}",
                        "max": f"time_index_max_{column}",
                        "min": f"time_index_min_{column}",
                    }
                )

        # course_lenは数値なので別途処理
        agg_result = (
            self.df_target_horses.sort_values(["horse_id", "date"])
            .groupby(["horse_id", "course_len"])["タイム指数"]
            .agg(["mean", "max", "min"])
            .reset_index()
        )
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            agg_result,
            on=["horse_id", "course_len"],
            how="left",
            suffixes=("", "_course_len"),
        )
        self.df_shutsuba_table = self.df_shutsuba_table.rename(
            columns={
                "mean": "time_index_mean_course_len",
                "max": "time_index_max_course_len",
                "min": "time_index_min_course_len",
            }
        )

        # 前回のレースからの間隔を計算（日数）
        last_race_dates = (
            self.df_target_horses.sort_values(["horse_id", "date"])
            .groupby("horse_id")["date"]
            .last()
            .reset_index()
        )
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            last_race_dates, on="horse_id", how="left", suffixes=("", "_last_race")
        )
        self.df_shutsuba_table["days_since_last_race"] = (
            pd.to_datetime(self.df_shutsuba_table["date"])
            - pd.to_datetime(self.df_shutsuba_table["date_last_race"])
        ).dt.days

        # 1レース前の情報を追加
        last_race_info = (
            self.df_target_horses.sort_values(["horse_id", "date"])
            .groupby("horse_id")
            .last()
            .reset_index()
        )
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            last_race_info[["horse_id", "タイム指数"]],
            on="horse_id",
            how="left",
            suffixes=("", "_prev"),
        )
        self.df_shutsuba_table = self.df_shutsuba_table.rename(
            columns={"タイム指数": "prev_タイム指数"}
        )

    def _add_season(self):
        # 季節を追加（春：3-5月、夏：6-8月、秋：9-11月、冬：12-2月）
        self.df_shutsuba_table["season"] = pd.Categorical(
            self.df_shutsuba_table["date"].dt.month.map(
                {
                    1: "冬",
                    2: "冬",
                    3: "春",
                    4: "春",
                    5: "春",
                    6: "夏",
                    7: "夏",
                    8: "夏",
                    9: "秋",
                    10: "秋",
                    11: "秋",
                    12: "冬",
                }
            ),
            categories=["春", "夏", "秋", "冬"],
            ordered=True,
        )

    def _add_chokyo_info(self):

        # Get the most common training location and trainer for each horse
        chokyo_info = (
            self.df_target_horses.groupby("horse_id")
            .agg(
                {
                    "調教場所": lambda x: (
                        x.mode().iloc[0] if not x.mode().empty else None
                    ),
                    "調教師名前": lambda x: (
                        x.mode().iloc[0] if not x.mode().empty else None
                    ),
                }
            )
            .reset_index()
        )

        # Merge the information with df_shutsuba_table
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            chokyo_info, on="horse_id", how="left", suffixes=("", "_most_common")
        )

        # カテゴリ変数の処理

    def _get_categorical_values(self):

        # そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
        # 列を一定にするため
        weathers = ["曇", "晴", "雨", "小雨", "小雪", "雪"]
        race_types = ["芝", "ダート", "障害"]
        ground_states = ["良", "稍", "重", "不"]
        sexes = ["牝", "牡", "セ"]
        places = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
        self.df_shutsuba_table["weather"] = pd.Categorical(
            self.df_shutsuba_table["weather"], weathers
        )
        self.df_shutsuba_table["race_type"] = pd.Categorical(
            self.df_shutsuba_table["race_type"], race_types
        )
        self.df_shutsuba_table["ground_state"] = pd.Categorical(
            self.df_shutsuba_table["ground_state"], ground_states
        )
        self.df_shutsuba_table["性"] = pd.Categorical(
            self.df_shutsuba_table["性"], sexes
        )
        self.df_shutsuba_table["place"] = pd.Categorical(
            self.df_shutsuba_table["place"], places
        )

        self.df_shutsuba_table = pd.get_dummies(
            self.df_shutsuba_table,
            columns=[
                "weather",
                "race_type",
                "ground_state",
                "性",
                "place",
                "season",
            ],
        )

    def _drop_and_rename_columns(self):
        self.df_shutsuba_table.drop(
            [
                "印",
                "馬名",
                "騎手",
                "厩舎",
                "馬体重(増減)",
                "性齢",
                "Unnamed:9_level_1",
                "人気",
                "登録",
                "メモ",
                "人気",
                "date_last_race",
            ],
            axis=1,
            inplace=True,
        )
        self.df_shutsuba_table.rename(columns={"枠": "枠番"}, inplace=True)
