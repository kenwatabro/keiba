import numpy as np
import pandas as pd
from abc import ABC


class DataProcessor(ABC):
    """
    Attributes:
    ----------
    data_raw : pd.DataFrame
        rawデータ
    data_preprocessed : pd.DataFrame
        preprocessing後のデータ
    data_horse_result_merged : pd.DataFrame
        merge_horse_results後のデータ
    data_peds_merged : pd.DataFrame
        merge_peds後のデータ
    data_categorical_processed : pd.DataFrame
        process_categorical後のデータ
    no_peds: Numpy.array
        merge_pedsを実行した時に、血統データが存在しなかった馬のhorse_id一覧
    """

    def __init__(self):
        self.data_raw_horseresults= pd.DataFrame()
        self.data_raw_horsepeds = pd.DataFrame()
        self.data_preprocessed = pd.DataFrame()


    # def merge_horse_results(self, hr, n_samples_list=[5, 9, "all"]):
    #     """
    #     馬の過去成績データから、
    #     n_samples_listで指定されたレース分の着順と賞金の平均を追加してdata_hに返す

    #     Parameters:
    #     ----------
    #     hr : HorseResults
    #         馬の過去成績データ
    #     n_samples_list : list, default [5, 9, 'all']
    #         過去何レース分追加するか
    #     """

    #     self.data_horse_result_merged = self.data_preprocessed.copy()
    #     for n_samples in n_samples_list:
    #         self.data_horse_result_merged = hr.merge_all(
    #             self.data_horse_result_merged, n_samples=n_samples
    #         )

    #     # 6/6追加： 馬の出走間隔追加
    #     self.data_horse_result_merged["interval"] = (
    #         self.data_horse_result_merged["date"]
    #         - self.data_horse_result_merged["latest"]
    #     ).dt.days
    #     self.data_horse_result_merged.drop(["開催", "latest"], axis=1, inplace=True)

    # def merge_peds(self, peds):
    #     """
    #     5世代分血統データを追加してdata_peに返す

    #     Parameters:
    #     ----------
    #     peds : Peds.peds_e
    #         Pedsクラスで加工された血統データ。
    #     """

    #     self.data_peds_merged = self.data_horse_result_merged.merge(
    #         peds, left_on="horse_id", right_index=True, how="left"
    #     )
    #     self.no_peds = self.data_peds_merged[self.data_peds_merged["peds_0"].isnull()][
    #         "horse_id"
    #     ].unique()
    #     if len(self.no_peds) > 0:
    #         print('scrape peds at horse_id_list "no_peds"')

    # def process_categorical(self, le_horse, le_jockey, results_m):
    #     """
    #     カテゴリ変数を処理してdata_cに返す

    #     Parameters:
    #     ----------
    #     le_horse : sklearn.preprocessing.LabelEncoder
    #         horse_idを0始まりの整数に変換するLabelEncoderオブジェクト。
    #     le_jockey : sklearn.preprocessing.LabelEncoder
    #         jockey_idを0始まりの整数に変換するLabelEncoderオブジェクト。
    #     results_m : Results.data_pe
    #         ダミー変数化のとき、ResultsクラスとShutubaTableクラスで列を合わせるためのもの
    #     """

    #     df = self.data_peds_merged.copy()

    #     # ラベルエンコーディング。horse_id, jockey_idを0始まりの整数に変換
    #     mask_horse = df["horse_id"].isin(le_horse.classes_)
    #     new_horse_id = df["horse_id"].mask(mask_horse).dropna().unique()
    #     le_horse.classes_ = np.concatenate([le_horse.classes_, new_horse_id])
    #     df["horse_id"] = le_horse.transform(df["horse_id"])
    #     mask_jockey = df["jockey_id"].isin(le_jockey.classes_)
    #     new_jockey_id = df["jockey_id"].mask(mask_jockey).dropna().unique()
    #     le_jockey.classes_ = np.concatenate([le_jockey.classes_, new_jockey_id])
    #     df["jockey_id"] = le_jockey.transform(df["jockey_id"])

    #     # horse_id, jockey_idをpandasのcategory型に変換
    #     df["horse_id"] = df["horse_id"].astype("category")
    #     df["jockey_id"] = df["jockey_id"].astype("category")

    #     # そのほかのカテゴリ変数をpandasのcategory型に変換してからダミー変数化
    #     # 列を一定にするため
    #     weathers = results_m["weather"].unique()
    #     race_types = results_m["race_type"].unique()
    #     ground_states = results_m["ground_state"].unique()
    #     sexes = results_m["性"].unique()
    #     df["weather"] = pd.Categorical(df["weather"], weathers)
    #     df["race_type"] = pd.Categorical(df["race_type"], race_types)
    #     df["ground_state"] = pd.Categorical(df["ground_state"], ground_states)
    #     df["性"] = pd.Categorical(df["性"], sexes)
    #     df = pd.get_dummies(df, columns=["weather", "race_type", "ground_state", "性"])

    #     self.data_categorical_processed = df
