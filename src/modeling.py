"""
modeling.py
===========
Chronological train/validation/test split, preprocessing pipelines, and
model definitions for the `is_high_engagement` classification task.
Direct refactor of notebooks/03_modeling.ipynb (Part 3, sections 1-7).

Target: is_high_engagement (top-quartile engagement_score; ~25% positive).
Features: the leakage-safe set assembled in feature_engineering.py, minus
word_count_bucket (dropped here only -- redundant with word_count /
log_word_count, which would double-count length information across a
numeric and a categorical column for tree models, and add uninformative
multicollinearity for the linear model).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "asker_prior_questions_count", "asker_prior_quality_rate",
    "topic_prior_count", "topic_prior_quality_rate",
    "word_count", "log_word_count", "word_count_z_within_topic",
    "asker_account_age_at_question_days",
]
CATEGORICAL_FEATURES = [
    "asker_country", "asker_device_type", "account_age_bucket", "topic",
    "created_dow", "created_is_weekend", "variant",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "is_high_engagement"


# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Train on the past, validate/test on data the model has never seen
    chronologically -- the realistic simulation of scoring future,
    not-yet-posted questions. A random split would leak information because
    several engineered features (e.g. word_count_z_within_topic) are
    computed from dataset-wide statistics.

    Returns (train_df, val_df, test_df), each sorted by created_date.
    """
    df = df.sort_values(["created_date", "question_id"]).reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def get_xy(df: pd.DataFrame):
    return df[FEATURES], df[TARGET]


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def build_preprocessors() -> tuple[ColumnTransformer, ColumnTransformer]:
    """
    Two preprocessors: tree models don't need scaling (scale-invariant),
    Logistic Regression gets an added StandardScaler. Numeric imputation is
    median-based, fit on train only. Categorical is one-hot with
    handle_unknown="ignore" so an unseen category at inference time doesn't
    break the pipeline.
    """
    preprocessor_tree = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    preprocessor_linear = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return preprocessor_tree, preprocessor_linear


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_models(y_train: pd.Series) -> dict[str, Pipeline]:
    """
    Four models, each chosen for a specific reason rather than "more models
    = better":
      1. Logistic Regression -- fast, fully interpretable baseline.
      2. Random Forest -- captures non-linear effects/interactions without
         manual feature crosses.
      3. HistGradientBoostingClassifier -- usually the strongest tabular
         performer; native missing-value handling.
      4. XGBoost -- included because the dataset (mixed numeric/categorical,
         moderate size, meaningful but non-trivial signal) is exactly the
         setting where gradient boosting variants tend to add value.

    All imbalance-sensitive hyperparameters use the TRAIN-set class ratio
    only (never validation/test information).
    """
    preprocessor_tree, preprocessor_linear = build_preprocessors()
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor_linear),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor_tree),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=10, min_samples_leaf=5,
                class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting (HGB)": Pipeline([
            ("pre", preprocessor_tree),
            ("clf", HistGradientBoostingClassifier(
                max_iter=300, max_depth=6, learning_rate=0.05,
                class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
    }
    try:
        import xgboost as xgb
        models["XGBoost"] = Pipeline([
            ("pre", preprocessor_tree),
            ("clf", xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, eval_metric="logloss")),
        ])
    except ImportError:
        pass  # xgboost is optional; the other three models still run.

    return models


def build_dummy_baseline() -> Pipeline:
    """Majority-class baseline. Sets the evaluation floor: ~75% accuracy
    with zero recall on the class that actually matters -- why accuracy is
    excluded from model selection here."""
    preprocessor_tree, _ = build_preprocessors()
    return Pipeline([("pre", preprocessor_tree), ("clf", DummyClassifier(strategy="most_frequent"))])


def fit_all(models: dict[str, Pipeline], X_train, y_train) -> dict[str, Pipeline]:
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
    return models


if __name__ == "__main__":
    df = pd.read_csv("../data/modeling_dataset_v2.csv", parse_dates=["created_date"])
    train_df, val_df, test_df = chronological_split(df)
    X_train, y_train = get_xy(train_df)
    X_val, y_val = get_xy(val_df)

    dummy = build_dummy_baseline()
    dummy.fit(X_train, y_train)

    models = build_models(y_train)
    fit_all(models, X_train, y_train)
    print(f"Trained {len(models)} models: {list(models.keys())}")
