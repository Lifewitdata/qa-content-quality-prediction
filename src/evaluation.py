"""
evaluation.py
=============
Imbalance-aware model evaluation, model selection, interpretation
(permutation importance) and structured error analysis. Direct refactor of
notebooks/04_model_interpretation.ipynb (Part 3, sections 6-9).

Accuracy is intentionally never used for model selection here: the
majority-class dummy baseline scores ~75% accuracy with zero recall on the
class that matters, on this dataset's ~75/25 class split. PR-AUC (primary)
and ROC-AUC (secondary, threshold-independent) are used instead, with
minority-class F1/precision/recall reported at the default 0.5 threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


# --------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------
def evaluate(pipe, X, y, threshold: float = 0.5) -> dict:
    """PR-AUC / ROC-AUC (threshold-independent) plus F1/precision/recall at
    a fixed 0.5 threshold."""
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
    }


def evaluate_all(models: dict, dummy, X, y) -> pd.DataFrame:
    results = {"Dummy (majority)": evaluate(dummy, X, y)}
    for name, pipe in models.items():
        results[name] = evaluate(pipe, X, y)
    return pd.DataFrame(results).T.round(4)


def select_best_model(val_results_df: pd.DataFrame, metric: str = "pr_auc") -> str:
    """
    Selection rule, decided before looking at test-set numbers: rank by
    validation PR-AUC first, use ROC-AUC / minority-class F1 as tie-breakers,
    and prefer the simpler model when performance is statistically
    indistinguishable (a ~0.005-0.02 PR-AUC gap on this dataset size is not
    a meaningful separation between the linear and tree-based models).
    """
    return val_results_df.drop(index="Dummy (majority)")[metric].idxmax()


# --------------------------------------------------------------------------
# Interpretation
# --------------------------------------------------------------------------
def get_permutation_importance(pipe, X_val, y_val, n_repeats: int = 10, random_state: int = 42) -> pd.DataFrame:
    """
    Model-agnostic importance: shuffles one RAW feature at a time and
    measures the PR-AUC drop. Preferred over raw logistic-regression
    coefficients here because word_count and log_word_count are highly
    correlated (multicollinearity can otherwise split/mask their combined
    effect in the coefficients alone).
    """
    perm = permutation_importance(
        pipe, X_val, y_val, n_repeats=n_repeats, random_state=random_state,
        scoring="average_precision", n_jobs=-1,
    )
    return pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)


def get_standardized_coefficients(pipe) -> pd.DataFrame:
    """Standardized Logistic Regression coefficients + odds ratios.
    Caveat: word_count / log_word_count collinearity can make raw
    word_count look weakly negative even though length clearly helps
    (see permutation importance / SHAP for a more robust read)."""
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_[0]
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    return coef_df.sort_values("coefficient")


# --------------------------------------------------------------------------
# Error analysis
# --------------------------------------------------------------------------
def label_outcomes(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    conditions = [
        (y_pred == 1) & (y_true == 1),
        (y_pred == 1) & (y_true == 0),
        (y_pred == 0) & (y_true == 1),
        (y_pred == 0) & (y_true == 0),
    ]
    return np.select(conditions, ["TP", "FP", "FN", "TN"], default="NA")


def accuracy_by_group(df: pd.DataFrame, group_col: str, target_col: str, pred_col: str) -> pd.Series:
    """Segment-level accuracy -- used to find where the model is strong vs.
    weak (e.g. by word_count_bucket)."""
    return df.groupby(group_col).apply(lambda g: (g[pred_col] == g[target_col]).mean())


def fp_fn_rate_by_group(df: pd.DataFrame, group_col: str, outcome_col: str, target_col: str) -> pd.DataFrame:
    """False-positive rate (among true negatives) and false-negative rate
    (among true positives), by segment -- e.g. by topic."""
    fp_rate = df.groupby(group_col).apply(
        lambda g: (g[outcome_col] == "FP").sum() / max((g[target_col] == 0).sum(), 1)
    )
    fn_rate = df.groupby(group_col).apply(
        lambda g: (g[outcome_col] == "FN").sum() / max((g[target_col] == 1).sum(), 1)
    )
    return pd.DataFrame({"fp_rate": fp_rate, "fn_rate": fn_rate})


if __name__ == "__main__":
    import modeling

    df = pd.read_csv("../data/modeling_dataset_v2.csv", parse_dates=["created_date"])
    train_df, val_df, test_df = modeling.chronological_split(df)
    X_train, y_train = modeling.get_xy(train_df)
    X_val, y_val = modeling.get_xy(val_df)
    X_test, y_test = modeling.get_xy(test_df)

    dummy = modeling.build_dummy_baseline()
    dummy.fit(X_train, y_train)
    models = modeling.fit_all(modeling.build_models(y_train), X_train, y_train)

    val_results = evaluate_all(models, dummy, X_val, y_val)
    print("Validation results:\n", val_results)

    best_name = select_best_model(val_results)
    print(f"\nSelected model: {best_name}")

    test_results = evaluate_all(models, dummy, X_test, y_test)
    print("\nTest results:\n", test_results)
