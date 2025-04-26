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
import optuna
from optuna.integration import lightgbm as lgb_o
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
import pickle

from modules.Preprocess import HorsePastPreprocessor

DATA_PATH = "horse_preprocessed/Race_preprocessed.pickle"


def get_data():
    if os.path.exists(DATA_PATH):
        data = pd.read_pickle(DATA_PATH)
    else:
        data = _preprocess_horse_data()
    return data


def _preprocess_horse_data():
    # ディレクトリが存在しない場合は作成
    os.makedirs("horse_preprocessed", exist_ok=True)
    
    df_race_result_all = pd.DataFrame()
    for file in os.listdir("horse_results"):
        if file.endswith(".pickle"):
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
    # 日付でソート
    df = df.sort_values('date')
    
    # 時系列を考慮した分割
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    return train, test


def select_features_by_importance(X_train, y_train, X_valid, y_valid, importance_threshold=0.01):
    # カテゴリカル変数の処理
    categorical_columns = ["性", "調教場所"]
    for col in categorical_columns:
        if col in X_train.columns:
            # 欠損値を'Unknown'に置換
            X_train[col] = X_train[col].fillna('Unknown')
            X_valid[col] = X_valid[col].fillna('Unknown')
            # エンコーディング
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            X_valid[col] = le.transform(X_valid[col])
            # カテゴリ型に変換
            X_train[col] = X_train[col].astype('category')
            X_valid[col] = X_valid[col].astype('category')
    
    # 初期モデルをトレーニング
    model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=100,
        random_state=42
    )
    
    # 特徴量の重要度を計算
    model.fit(X_train, y_train)
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })
    
    # 重要度の高い特徴量を選択
    important_features = importance[importance['importance'] > importance_threshold]['feature']
    
    return important_features.tolist()

def optimize_hyperparameters(X_train, y_train, X_valid, y_valid):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
        }
        
        lgb_train = lgb_o.Dataset(X_train.values, y_train.values)
        lgb_valid = lgb_o.Dataset(X_valid.values, y_valid.values)
        
        model = lgb_o.train(
            params,
            lgb_train,
            num_boost_round=1000,
            valid_sets=(lgb_train, lgb_valid),
            callbacks=[
                lgb_o.early_stopping(stopping_rounds=50),
                lgb_o.log_evaluation(period=100),
            ],
            optuna_seed=100,
        )
        
        return model.best_score['valid_0']['auc']
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)
    
    return study.best_params

def train_model(data, target="rank"):
    # データの分割（時系列を考慮）
    train, test = split_data(data)
    
    # カテゴリカル変数のエンコーディング（トレーニングデータのみで学習）
    categorical_columns = ["性", "調教場所"]
    encoders = {}
    
    # トレーニングデータのみでエンコーダーを学習
    for col in categorical_columns:
        if col in train.columns:
            encoders[col] = LabelEncoder()
            # 欠損値を'Unknown'に置換
            train[col] = train[col].fillna('Unknown')
            test[col] = test[col].fillna('Unknown')
            # エンコーディング
            train[col] = encoders[col].fit_transform(train[col])
            # テストデータは学習済みのエンコーダーで変換
            test[col] = test[col].map(lambda x: x if x in encoders[col].classes_ else "Unknown")
            test[col] = encoders[col].transform(test[col])
            # カテゴリ型に変換
            train[col] = train[col].astype('category')
            test[col] = test[col].astype('category')
    
    # 日付関連の特徴量を作成
    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["day"] = train["date"].dt.day
    train["dayofweek"] = train["date"].dt.dayofweek
    
    test["year"] = test["date"].dt.year
    test["month"] = test["date"].dt.month
    test["day"] = test["date"].dt.day
    test["dayofweek"] = test["date"].dt.dayofweek
    
    # 日付列を削除
    train = train.drop("date", axis=1)
    test = test.drop("date", axis=1)
    
    # 特徴量とターゲットの分離
    drop_cols = [col for col in [target, "単勝的中", "単勝オッズ"] if col in train.columns]
    X_train = train.drop(drop_cols, axis=1)
    y_train = train[target]
    X_test = test.drop(drop_cols, axis=1)
    y_test = test[target]
    
    # 重要度の高い特徴量を選択（トレーニングデータのみで選択）
    important_features = select_features_by_importance(X_train, y_train, X_train, y_train)
    X_train = X_train[important_features]
    X_test = X_test[important_features]
    
    # ハイパーパラメータの最適化
    best_params = optimize_hyperparameters(X_train, y_train, X_train, y_train)
    
    # モデルの学習
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # モデルとエンコーダーを保存
    os.makedirs("trained_models", exist_ok=True)
    with open("trained_models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("trained_models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open("trained_models/important_features.pkl", "wb") as f:
        pickle.dump(important_features, f)
    
    # モデルの評価
    y_pred = model.predict(X_test)
    auc_score = roc_auc_score(y_test, y_pred)
    print(f"Test AUC: {auc_score:.4f}")
    
    return model, important_features, encoders

if __name__ == "__main__":
    data = get_data()
    data = data.drop(columns=["タイム指数", "上り"])
    train, test = split_data(data)
    train, valid = split_data(train)

    X_train = train.drop(["rank", "date"], axis=1)
    y_train = train["rank"]
    X_valid = valid.drop(["rank", "date"], axis=1)
    y_valid = valid["rank"]
    X_test = test.drop(["rank", "date"], axis=1)
    y_test = test["rank"]

    # 特徴量の選択
    important_features = select_features_by_importance(X_train, y_train, X_valid, y_valid)
    X_train = X_train[important_features]
    X_valid = X_valid[important_features]
    X_test = X_test[important_features]

    # ハイパーパラメータの最適化
    best_params = optimize_hyperparameters(X_train, y_train, X_valid, y_valid)
    
    # 最適化されたパラメータでモデルをトレーニング
    lgb_train = lgb_o.Dataset(X_train.values, y_train.values)
    lgb_valid = lgb_o.Dataset(X_valid.values, y_valid.values)
    lgb_test = lgb_o.Dataset(X_test.values, y_test.values)

    lgb_clf_o = lgb_o.train(
        best_params,
        lgb_train,
        num_boost_round=1000,
        valid_sets=(lgb_train, lgb_valid),
        callbacks=[
            lgb_o.early_stopping(stopping_rounds=50),
            lgb_o.log_evaluation(period=100),
        ],
        optuna_seed=100,
    )
    
    # 特徴量の重要度を可視化
    importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': lgb_clf_o.feature_importance()
    })
    importance = importance.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    plt.bar(importance['feature'][:20], importance['importance'][:20])
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
