"""
Loan Default Prediction – Model Training Script

Note:
- This code was originally built for a private / competition dataset.
- The dataset itself is NOT included in this repository.
- To run this script, replace the file paths and column names
  with those from your own dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

TRAIN_PATH = "data/train_dataset.csv"
TEST_PATH = "data/test_without_gt.csv"
SUB_TEMPLATE_PATH = "data/submission_template.csv"
OUTPUT_PATH = "submission.csv"

TARGET_COL = "default_12month"

DROP_COLS = [
    "ID",
    "postal_code",
    "c_postal_code",
    "date_of_birth",
    "APP_date",
]


def load_data(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    return train_df, test_df


def basic_cleanup(train_df, test_df):
    train_df = train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], errors="ignore")
    test_df = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns], errors="ignore")
    return train_df, test_df


def add_features(train_df, test_df):
    for df in (train_df, test_df):
        if {"living_period_year", "living_period_month"}.issubset(df.columns):
            df["living_period_total_months"] = df["living_period_year"] * 12 + df["living_period_month"]

        if {"c_number_of_working_year", "c_number_of_working_month"}.issubset(df.columns):
            df["work_experience_total_months"] = (
                df["c_number_of_working_year"] * 12 + df["c_number_of_working_month"]
            )

        if {"c_monthly_salary", "r_expected_credit_limit"}.issubset(df.columns):
            df["income_to_credit_ratio"] = df["c_monthly_salary"] / (df["r_expected_credit_limit"] + 1)

        if {"c_monthly_salary", "r_additional_income", "r_spouse_income"}.issubset(df.columns):
            df["total_income"] = (
                df["c_monthly_salary"] + df["r_additional_income"] + df["r_spouse_income"]
            )

        if {"r_allloan_amount", "total_income"}.issubset(df.columns):
            df["debt_to_income_ratio"] = df["r_allloan_amount"] / (df["total_income"] + 1)

    return train_df, test_df


def encode_categoricals(train_df, test_df, target_col):
    features = [col for col in train_df.columns if col != target_col]
    object_cols = [col for col in features if train_df[col].dtype == "object"]

    for col in object_cols:
        le = LabelEncoder()
        le.fit(list(train_df[col].astype(str)) + list(test_df[col].astype(str)))
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test[col].astype(str))

    return train_df, test_df


def train_lightgbm_cv(train_df, test_df, target_col):
    features = [col for col in train_df.columns if col != target_col]
    X = train_df[features]
    y = train_df[target_col]

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(test_df))

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        print(f"Fold {fold}")
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMClassifier(**params, n_estimators=5000)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="auc",
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(200),
            ],
        )

        oof_preds[valid_idx] = model.predict_proba(X_valid)[:, 1]
        test_preds += model.predict_proba(test_df)[:, 1] / folds.n_splits

    auc = roc_auc_score(y, oof_preds)
    print(f"Cross-validated AUC: {auc:.5f}")
    return oof_preds, test_preds, auc


def save_submission(test_preds, template_path, output_path, target_col):
    submission = pd.read_csv(template_path)
    submission[target_col] = test_preds
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path}")


if __name__ == "__main__":
    train, test = load_data(TRAIN_PATH, TEST_PATH)
    train, test = basic_cleanup(train, test)
    train, test = add_features(train, test)

    train = train.fillna(-1)
    test = test.fillna(-1)

    train, test = encode_categoricals(train, test, TARGET_COL)
    _, test_predictions, auc_score = train_lightgbm_cv(train, test, TARGET_COL)

    save_submission(test_predictions, SUB_TEMPLATE_PATH, OUTPUT_PATH, TARGET_COL)
