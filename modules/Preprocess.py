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
        self._add_new_features()
        self._add_interaction_features()
        self._enhance_time_series_features()
        self._handle_categorical_encoding()
        self._add_time_index_features()
        self._select_features()
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
        self.df_race_results["性"] = self.df_race_results["性齢"].str[0]
        self.df_race_results["年齢"] = self.df_race_results["性齢"].str[1:].astype(int)

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
        self.df_race_results["調教場所"] = self.df_race_results["調教師"].str[1:2]
        self.df_race_results["調教師名前"] = self.df_race_results["調教師"].str[3:]

    def _process_dates(self):
        # ['勝馬投票100周年' '三年坂ステークス' '競馬法100周年記念' '府中市70周年記念' '宝塚市制70周年記念']は個別処理
        event_mapping = {
            "勝馬投票100周年": "2023/08/27",
            "三年坂ステークス": "2023/10/22",
            "競馬法100周年記念": "2023/04/09",
            "府中市70周年記念": "2024/04/27",
            "宝塚市制70周年記念": "2024/04/07",
        }
        self.df_race_results["date"] = self.df_race_results["date"].replace(event_mapping)
        self.df_race_results["date"] = pd.to_datetime(
            self.df_race_results["date"].str.split("(").str[0]
        )

    def _process_pass_columns(self):
        pass_split = self.df_race_results["通過"].str.split("-", expand=True)
        pass_split = pass_split.iloc[:, :4]
        pass_split.columns = ["通過1", "通過2", "通過3", "通過4"]
        self.df_race_results = pd.concat([self.df_race_results, pass_split], axis=1)

        self.df_race_results["頭数"] = self.df_race_results.groupby("horse_id")["馬番"].transform("max")

        def calculate_run_type(pass1, total):
            if pd.isna(pass1):
                return None
            return int(pass1) / total

        self.df_race_results["run_type"] = self.df_race_results.apply(
            lambda row: calculate_run_type(row["通過1"], row["頭数"]), axis=1
        )

    def _process_agari(self):
        # Convert '上り' to numeric, replacing '-' with NaN
        self.df_race_results["上り"] = pd.to_numeric(
            self.df_race_results["上り"].replace("-", np.nan), errors="coerce"
        )

        # Sort by horse_id and date
        self.df_race_results = self.df_race_results.sort_values(["horse_id", "date"])

        # Calculate the rolling mean of '上り' for the last 5 races, excluding the current race
        self.df_race_results["avg_上り_last_5"] = self.df_race_results.groupby("horse_id")[
            "上り"
        ].transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())

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

    def _add_new_features(self):
        # 1. 馬の調子指標
        self.df_race_results["調子スコア"] = (
            self.df_race_results["体重変化"].fillna(0) * 0.3 +
            self.df_race_results["上り"].fillna(0) * 0.7
        )
        
        # 2. レース間隔
        self.df_race_results = self.df_race_results.sort_values(["horse_id", "date"])
        self.df_race_results["前回レースからの日数"] = (
            self.df_race_results.groupby("horse_id")["date"]
            .diff()
            .dt.days
        )
        
        # 3. 過去の成績指標
        self.df_race_results["過去5走の着順平均"] = (
            self.df_race_results.groupby("horse_id")["着順"]
            .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
        )
        
        # 4. コース適性
        self.df_race_results["コース適性スコア"] = (
            self.df_race_results.groupby(["horse_id", "race_type"])["タイム指数"]
            .transform("mean")
        )
        
        # 5. 騎手の調子
        self.df_race_results["騎手の調子"] = (
            self.df_race_results.groupby("jockey_id")["タイム指数"]
            .transform(lambda x: x.shift().rolling(window=5, min_periods=1).mean())
        )

    def _add_interaction_features(self):
        # 1. 年齢と体重の交互作用
        self.df_race_results["年齢体重スコア"] = (
            self.df_race_results["年齢"] * self.df_race_results["体重"]
        )
        
        # 2. コースと天気の交互作用
        weather_dummies = pd.get_dummies(self.df_race_results["weather"], prefix="weather")
        self.df_race_results["コース天気スコア"] = (
            self.df_race_results["course_len"].values.reshape(-1, 1) * 
            weather_dummies.values
        )
        
        # 3. 馬場状態と走法の交互作用
        ground_dummies = pd.get_dummies(self.df_race_results["ground_state"], prefix="ground")
        self.df_race_results["馬場走法スコア"] = (
            self.df_race_results["run_type"].values.reshape(-1, 1) * 
            ground_dummies.values
        )

    def _enhance_time_series_features(self):
        # 1. より長期のトレンド
        for window in [10, 20, 30]:
            self.df_race_results[f"タイム指数トレンド_{window}"] = (
                self.df_race_results.groupby("horse_id")["タイム指数"]
                .transform(lambda x: x.shift().rolling(window=window, min_periods=1).mean())
            )
        
        # 2. 季節性の考慮
        self.df_race_results["月別平均タイム"] = (
            self.df_race_results.groupby(["horse_id", "date"].dt.month)["タイム指数"]
            .transform("mean")
        )
        
        # 3. レース間隔の統計量
        self.df_race_results["レース間隔の標準偏差"] = (
            self.df_race_results.groupby("horse_id")["前回レースからの日数"]
            .transform("std")
        )

    def _handle_categorical_encoding(self):
        # 一貫したエンコーディング手法の適用
        low_cardinality_cols = ["weather", "race_type", "ground_state", "season"]
        high_cardinality_cols = ["horse_id", "jockey_id", "調教師名前"]

        # ワンホットエンコーディング
        self.df_race_results = pd.get_dummies(
            self.df_race_results,
            columns=low_cardinality_cols,
            drop_first=True
        )

        # ラベルエンコーディング
        for col in high_cardinality_cols:
            le_path = os.path.join(JOBLIB_PATH, f"le_{col}.joblib")
            if os.path.exists(le_path):
                le = joblib.load(le_path)
            else:
                # ディレクトリが存在しない場合は作成
                os.makedirs(JOBLIB_PATH, exist_ok=True)
                le = LabelEncoder()
                le.fit(self.df_race_results[col])
                joblib.dump(le, le_path)
            # 未知のカテゴリは -1 に設定
            self.df_race_results[col] = self.df_race_results[col].map(lambda x: x if x in le.classes_ else "Unknown")
            le_classes = np.append(le.classes_, "Unknown")
            le.classes_ = le_classes
            self.df_race_results[col] = le.transform(self.df_race_results[col])
            self.df_race_results[col] = self.df_race_results[col].astype("category")

    def _add_time_index_features(self):
        # 日付でソート
        self.df_race_results = self.df_race_results.sort_values(["horse_id", "date"])

        # 固定ウィンドウを使用してローリングフィーチャーを計算
        window_sizes = [3, 5]
        for window in window_sizes:
            self.df_race_results[f"time_index_mean_last_{window}"] = self.df_race_results.groupby("horse_id")["タイム指数"].transform(
                lambda x: x.shift().rolling(window=window, min_periods=1).mean()
            )
            self.df_race_results[f"time_index_max_last_{window}"] = self.df_race_results.groupby("horse_id")["タイム指数"].transform(
                lambda x: x.shift().rolling(window=window, min_periods=1).max()
            )
            self.df_race_results[f"time_index_min_last_{window}"] = self.df_race_results.groupby("horse_id")["タイム指数"].transform(
                lambda x: x.shift().rolling(window=window, min_periods=1).min()
            )

    def _select_features(self):
        # 1. 相関の高い特徴量の削除
        correlation_threshold = 0.95
        corr_matrix = self.df_race_results.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        self.df_race_results.drop(columns=to_drop, inplace=True)
        
        # 2. 重要度の低い特徴量の削除（トレーニング時に実行）
        # この部分は train.py で実装することを推奨

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
        self._handle_categorical_encoding()
        self._add_time_index_features()
        self._drop_and_rename_columns()
        return self.df_shutsuba_table

    def _convert_ids_to_category(self):
        # 一貫したエンコーディング手法を適用
        categorical_cols = ["horse_id", "jockey_id"]

        for col in categorical_cols:
            le_path = os.path.join(JOBLIB_PATH, f"le_{col}.joblib")
            if os.path.exists(le_path):
                le = joblib.load(le_path)
            else:
                le = LabelEncoder()
                le.fit(self.df_shutsuba_table[col])
                joblib.dump(le, le_path)
            # 未知のカテゴリは "Unknown" に設定
            self.df_shutsuba_table[col] = self.df_shutsuba_table[col].map(
                lambda x: x if x in le.classes_ else "Unknown"
            )
            le_classes = np.append(le.classes_, "Unknown")
            le.classes_ = le_classes
            self.df_shutsuba_table[col] = le.transform(self.df_shutsuba_table[col])
            self.df_shutsuba_table[col] = self.df_shutsuba_table[col].astype("category")

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
        self.df_shutsuba_table["性"] = self.df_shutsuba_table["性齢"].str[0]
        self.df_shutsuba_table["年齢"] = self.df_shutsuba_table["性齢"].str[1:].astype(int)

        # 馬体重を体重と体重変化に分ける
        self.df_shutsuba_table["体重"] = self.df_shutsuba_table["馬体重(増減)"].str.split("(", expand=True)[0]
        self.df_shutsuba_table["体重変化"] = (
            self.df_shutsuba_table["馬体重(増減)"].str.split("(", expand=True)[1].str[:-1]
        )

        # errors='coerce'で、"計不"など変換できない時に欠損値にする
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
        avg_run_type = self.df_target_horses.groupby("horse_id")["run_type"].mean()
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            avg_run_type.rename("run_type"), on="horse_id", how="left"
        )

    def _get_estimated_agari(self):
        avg_agari = self.df_target_horses.groupby("horse_id")["上り"].mean()
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            avg_agari.rename("avg_上り"), on="horse_id", how="left"
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
        # 調教情報を追加
        chokyo_info = self.df_target_horses.groupby("horse_id").agg(
            {
                "調教場所": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
                "調教師名前": lambda x: x.mode().iloc[0] if not x.mode().empty else None,
            }
        ).reset_index()
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            chokyo_info, on="horse_id", how="left", suffixes=("", "_most_common")
        )

    def _handle_categorical_encoding(self):
        # 一貫したエンコーディング手法の適用
        low_cardinality_cols = ["weather", "race_type", "ground_state", "season"]
        high_cardinality_cols = ["horse_id", "jockey_id", "調教師名前"]

        # ワンホットエンコーディング
        self.df_shutsuba_table = pd.get_dummies(
            self.df_shutsuba_table,
            columns=low_cardinality_cols,
            drop_first=True
        )

        # ラベルエンコーディング
        for col in high_cardinality_cols:
            le_path = os.path.join(JOBLIB_PATH, f"le_{col}.joblib")
            if os.path.exists(le_path):
                le = joblib.load(le_path)
            else:
                le = LabelEncoder()
                le.fit(self.df_shutsuba_table[col])
                joblib.dump(le, le_path)
            # 未知のカテゴリは "Unknown" に設定
            self.df_shutsuba_table[col] = self.df_shutsuba_table[col].map(lambda x: x if x in le.classes_ else "Unknown")
            le_classes = np.append(le.classes_, "Unknown")
            le.classes_ = le_classes
            self.df_shutsuba_table[col] = le.transform(self.df_shutsuba_table[col])
            self.df_shutsuba_table[col] = self.df_shutsuba_table[col].astype("category")

    def _add_time_index_features(self):
        # 日付でソート
        self.df_shutsuba_table = self.df_shutsuba_table.sort_values(["horse_id", "date"])

        # 固定ウィンドウを使用してローリングフィーチャーを計算
        window_sizes = [3, 5]
        for window in window_sizes:
            self.df_shutsuba_table[f"time_index_mean_last_{window}"] = self.df_shutsuba_table.groupby("horse_id")["タイム指数"].transform(
                lambda x: x.shift().rolling(window=window, min_periods=1).mean()
            )
            self.df_shutsuba_table[f"time_index_max_last_{window}"] = self.df_shutsuba_table.groupby("horse_id")["タイム指数"].transform(
                lambda x: x.shift().rolling(window=window, min_periods=1).max()
            )
            self.df_shutsuba_table[f"time_index_min_last_{window}"] = self.df_shutsuba_table.groupby("horse_id")["タイム指数"].transform(
                lambda x: x.shift().rolling(window=window, min_periods=1).min()
            )

        # 条件別の集計フィーチャー
        condition_columns = {
            "weather": [
                "weather_曇", "weather_晴", "weather_雨", "weather_小雨", "weather_小雪", "weather_雪",
            ],
            "race_type": ["race_type_芝", "race_type_ダート", "race_type_障害"],
            "ground_state": ["ground_state_良", "ground_state_稍", "ground_state_重", "ground_state_不"],
            "place": [
                "place_01", "place_02", "place_03", "place_04", "place_05",
                "place_06", "place_07", "place_08", "place_09", "place_10",
            ],
        }

        for condition, columns in condition_columns.items():
            for column in columns:
                agg_result = self.df_target_horses[self.df_target_horses[column] == 1].groupby("horse_id")["タイム指数"].agg(["mean", "max", "min"]).reset_index()
                self.df_shutsuba_table = self.df_shutsuba_table.merge(
                    agg_result, on="horse_id", how="left", suffixes=("", f"_{column}")
                )
                self.df_shutsuba_table.rename(
                    columns={
                        "mean": f"time_index_mean_{column}",
                        "max": f"time_index_max_{column}",
                        "min": f"time_index_min_{column}",
                    },
                    inplace=True
                )

        # Course length の集計
        agg_result = self.df_target_horses.groupby(["horse_id", "course_len"])["タイム指数"].agg(["mean", "max", "min"]).reset_index()
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            agg_result, on=["horse_id", "course_len"], how="left", suffixes=("", "_course_len")
        )
        self.df_shutsuba_table.rename(
            columns={
                "mean": "time_index_mean_course_len",
                "max": "time_index_max_course_len",
                "min": "time_index_min_course_len",
            },
            inplace=True
        )

        # 前回のレースからの間隔を計算
        last_race_dates = self.df_target_horses.groupby("horse_id")["date"].max().reset_index()
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            last_race_dates.rename(columns={"date": "last_race_date"}), on="horse_id", how="left"
        )
        self.df_shutsuba_table["days_since_last_race"] = (
            pd.to_datetime(self.df_shutsuba_table["date"]) - pd.to_datetime(self.df_shutsuba_table["last_race_date"])
        ).dt.days

        # 1レース前のタイム指数を追加
        last_race_info = self.df_target_horses.groupby("horse_id").last().reset_index()
        self.df_shutsuba_table = self.df_shutsuba_table.merge(
            last_race_info[["horse_id", "タイム指数"]].rename(columns={"タイム指数": "prev_タイム指数"}),
            on="horse_id",
            how="left"
        )

    def _drop_and_rename_columns(self):
        columns_to_drop = [
            "印", "馬名", "騎手", "厩舎", "馬体重(増減)",
            "性齢", "Unnamed:9_level_1", "人気", "登録",
            "メモ", "last_race_date",
        ]
        self.df_shutsuba_table.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        self.df_shutsuba_table.rename(columns={"枠": "枠番"}, inplace=True)
