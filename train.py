import os
import pickle
import datetime
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import lightgbm as lgb
import optuna

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

from modules.Preprocess import HorsePastPreprocessor

# -----------------------------------------------------------------------------
# 0.  グローバル設定
# -----------------------------------------------------------------------------
DATA_PATH = "horse_preprocessed/Race_preprocessed.pickle"
CATEGORICAL_COLUMNS = ["性", "調教場所"]

# ----- 日本語フォントの設定 --------------------------------------------------
if not any("IPAGothic" in f.name for f in font_manager.fontManager.ttflist):
    # IPA ゴシックが無い環境 → Noto Sans CJK にフォールバック
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
else:
    plt.rcParams["font.family"] = "IPAGothic"
plt.rcParams["axes.unicode_minus"] = False


# -----------------------------------------------------------------------------
# 1.  データ取得・前処理
# -----------------------------------------------------------------------------

def get_data() -> pd.DataFrame:
    """Pickle があれば読み込み、無ければ前処理を実行"""
    if os.path.exists(DATA_PATH):
        return pd.read_pickle(DATA_PATH)
    return _preprocess_horse_data()


def _preprocess_horse_data() -> pd.DataFrame:
    os.makedirs("horse_preprocessed", exist_ok=True)

    df_all = []
    for file in os.listdir("horse_results"):
        if file.endswith(".pickle"):
            df = pd.read_pickle(f"horse_results/{file}")
            if not df.empty:
                df_all.append(df)
    df_race = pd.concat(df_all, ignore_index=True)

    hpp = HorsePastPreprocessor(df_race)
    df_processed = hpp.preprocess()
    df_processed.to_pickle(DATA_PATH)
    return df_processed


# -----------------------------------------------------------------------------
# 2.  ユーティリティ
# -----------------------------------------------------------------------------

def split_data(df: pd.DataFrame, test_size: float = 0.3):
    """時系列順に train / test を hold‑out 分割"""
    df_sorted = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()
    return train, test


def select_features_by_importance(X_train, y_train, X_valid, y_valid, top_k: int = 30):
    """LightGBM の gain で上位 top_k 特徴量を選択"""

    # --- カテゴリ変数エンコード（学習用簡易モデルなので直接 fit） ---
    X_train_enc, X_valid_enc = X_train.copy(), X_valid.copy()
    # --- CATEGORICAL_COLUMNS をエンコード ------------------------------
    for col in CATEGORICAL_COLUMNS:
        if col in X_train_enc.columns:
            # ⚠️ category → object に剥がしてから欠損補完
            X_train_enc[col] = X_train_enc[col].astype("object").fillna("Unknown")
            X_valid_enc[col] = X_valid_enc[col].astype("object").fillna("Unknown")

            le = LabelEncoder()
            X_train_enc[col] = le.fit_transform(X_train_enc[col])
            X_valid_enc[col] = le.transform(X_valid_enc[col])

            # LightGBM に渡すときは “整数 + Categorical” が最速
            X_train_enc[col] = pd.Categorical(X_train_enc[col])
            X_valid_enc[col] = pd.Categorical(X_valid_enc[col])


    lgb_tmp = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        random_state=42,
        verbose=-1,
    )
    lgb_tmp.fit(X_train_enc, y_train, eval_set=[(X_valid_enc, y_valid)], callbacks=[lgb.log_evaluation(0)])

    gain_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "gain": lgb_tmp.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)

    return gain_df["feature"].head(top_k).tolist()


# -----------------------------------------------------------------------------
# 3.  ハイパーパラメータ最適化
# -----------------------------------------------------------------------------

def optimize_hyperparameters(X_train, y_train, X_valid, y_valid, n_trials: int = 20):
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 1e-1, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[lgb.early_stopping(20)],
        )
        return model.best_score_["valid_0"]["binary_logloss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study.best_params


# -----------------------------------------------------------------------------
# 4.  メイン学習関数
# -----------------------------------------------------------------------------

def train_model(
    data: pd.DataFrame,
    target: str = "rank",
    test_size: float = 0.3,
    valid_size: float = 0.2,
):
    """時系列 hold‑out → train/valid/test の 3 分割で学習"""

    # ----- 時系列 hold‑out -----
    train_full, test = split_data(data, test_size=test_size)
    train, valid = split_data(train_full, test_size=valid_size)

    # ----- 日付特徴量 -----
    for df_ in (train, valid, test):
        df_["year"] = df_["date"].dt.year
        df_["month"] = df_["date"].dt.month
        df_["day"] = df_["date"].dt.day
        df_["dow"] = df_["date"].dt.dayofweek
        df_.drop(columns=["date"], inplace=True)

    # ----- X / y 分離 -----
    drop_cols = [c for c in (target, "単勝的中", "単勝オッズ") if c in train.columns]
    X_train, y_train = train.drop(columns=drop_cols), train[target]
    X_valid, y_valid = valid.drop(columns=drop_cols), valid[target]
    X_test, y_test = test.drop(columns=drop_cols), test[target]

    # ----- Feature selection -----
    feat_list = select_features_by_importance(
        X_train.copy(), y_train, X_valid.copy(), y_valid, top_k=30
    )
    X_train, X_valid, X_test = (
        X_train[feat_list], X_valid[feat_list], X_test[feat_list]
    )

    encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in X_train.columns:
            le = LabelEncoder()
            for df_ in (X_train, X_valid, X_test):
                df_[col] = df_[col].fillna("Unknown")
            X_train[col] = le.fit_transform(X_train[col])
            X_valid[col] = le.transform(X_valid[col])
            X_test[col]  = X_test[col].map(
                lambda x: x if x in le.classes_ else "Unknown"
            ).pipe(le.transform)
            for df_ in (X_train, X_valid, X_test):
                df_[col] = pd.Categorical(df_[col])
            encoders[col] = le

    # ----- Hyper‑parameter tuning -----
    best_params = optimize_hyperparameters(X_train, y_train, X_valid, y_valid)
    best_params["verbose"] = -1

    # ----- Final model: train+valid で再学習 -----
    X_train_full = pd.concat([X_train, X_valid], axis=0)
    y_train_full = pd.concat([y_train, y_valid], axis=0)

    model = lgb.LGBMClassifier(**best_params)
    model.fit(X_train_full, y_train_full)

    # ----- 評価 -----
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC: {auc:.4f}")

    # ----- 保存 -----
    os.makedirs("trained_models", exist_ok=True)
    with open("trained_models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("trained_models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open("trained_models/important_features.pkl", "wb") as f:
        pickle.dump(feat_list, f)

    # ----- Feature importance plot -----
    imp_df = pd.DataFrame(
        {
            "feature": feat_list,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    ).sort_values("gain", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(imp_df["feature"].head(20), imp_df["gain"].head(20))
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 20 Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    return model, feat_list, encoders, auc


# -----------------------------------------------------------------------------
# 5.  エントリポイント
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    raw_data = get_data()
    # 使わない列を削除（存在しない場合も無視）
    raw_data = raw_data.drop(columns=["タイム指数", "上り"], errors="ignore")

    model, feats, encoders, auc_score = train_model(raw_data, target="rank")
    print("\n=== Training finished ===")
    print(f"Selected {len(feats)} features → trained model saved to trained_models/ .")
