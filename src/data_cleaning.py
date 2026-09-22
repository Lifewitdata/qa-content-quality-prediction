"""
data_cleaning.py
=================
Loading, quality-checking, and cleaning of the four raw Q&A platform
extracts (users, questions, user_activity, experiment), and construction
of the leakage-safe, question-level modeling table.

This module is a direct refactor of the logic developed interactively in
notebooks/01_data_profiling.ipynb. It performs no new analysis — it
packages the already-validated cleaning and feature steps into reusable
functions so they can be re-run on a fresh data drop without re-deriving
the logic from scratch.

Grain: one row per `question_id`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_raw(data_dir: str = ".") -> dict[str, pd.DataFrame]:
    """Load the four raw CSV extracts."""
    users = pd.read_csv(f"{data_dir}/users.csv")
    questions = pd.read_csv(f"{data_dir}/questions.csv")
    activity = pd.read_csv(f"{data_dir}/user_activity.csv")
    experiment = pd.read_csv(f"{data_dir}/experiment.csv")
    return {"users": users, "questions": questions, "user_activity": activity, "experiment": experiment}


# --------------------------------------------------------------------------
# Data-quality checks (referential integrity, duplicates, invalid values)
# --------------------------------------------------------------------------
def check_referential_integrity(raw: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Counts of orphan foreign keys across the four tables. All expected to be 0."""
    users, questions, activity, experiment = (
        raw["users"], raw["questions"], raw["user_activity"], raw["experiment"]
    )
    return {
        "orphan_question_users": len(set(questions.user_id) - set(users.user_id)),
        "orphan_activity_users": len(set(activity.user_id) - set(users.user_id)),
        "orphan_activity_questions": len(set(activity.question_id) - set(questions.question_id)),
        "orphan_experiment_users": len(set(experiment.user_id) - set(users.user_id)),
        "users_missing_experiment_row": len(set(users.user_id) - set(experiment.user_id)),
    }


def check_activity_vs_questions_counters(activity: pd.DataFrame, questions: pd.DataFrame) -> pd.Series:
    """
    Compares questions.csv's own engagement counters against an independent
    aggregation of user_activity.csv. Finding (Part 1): these do NOT match —
    user_activity.csv is an independent, partial event log, not the source
    of questions.csv's summary counters. Both are kept as separate signals.
    """
    event_counts = (
        activity.groupby(["question_id", "activity_type"]).size().unstack(fill_value=0)
        .rename(columns={"view": "views_from_log", "answer": "answers_from_log",
                          "upvote": "upvotes_from_log", "ask": "asks_from_log"})
    )
    check = questions.merge(event_counts, on="question_id", how="left")
    return pd.Series({
        "num_views_match_rate": (check.num_views == check.views_from_log).mean(),
        "num_answers_match_rate": (check.num_answers == check.answers_from_log).mean(),
        "num_upvotes_match_rate": (check.num_upvotes == check.upvotes_from_log).mean(),
    })


def check_invalid_values(questions: pd.DataFrame, experiment: pd.DataFrame) -> pd.Series:
    """Out-of-range / logically-impossible value checks. All expected to be 0."""
    checks = {
        "word_count_lt_1": (questions.word_count < 1).sum(),
        "num_views_lt_0": (questions.num_views < 0).sum(),
        "num_answers_lt_0": (questions.num_answers < 0).sum(),
        "num_upvotes_lt_0": (questions.num_upvotes < 0).sum(),
        "got_quality_answer_not_binary": (~questions.got_quality_answer.isin([0, 1])).sum(),
        "num_answers_gt_num_views": (questions.num_answers > questions.num_views).sum(),
        "zero_answers_but_quality_flag_set": (
            (questions.num_answers == 0) & (questions.got_quality_answer == 1)
        ).sum(),
        "experiment_num_questions_lt_0": (experiment.num_questions_asked < 0).sum(),
    }
    return pd.Series(checks)


def iqr_outlier_count(series: pd.Series, k: float = 1.5) -> int:
    """Count of IQR-method outliers in a numeric series."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return int(((series < lo) | (series > hi)).sum())


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean_raw(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Standardize category text, parse dates, and encode structural missingness.
    No rows are dropped: Part 1 found no duplicate IDs, no orphaned foreign
    keys, and no invalid values in any of the four extracts.
    """
    q = raw["questions"].copy()
    u = raw["users"].copy()
    e = raw["experiment"].copy()
    a = raw["user_activity"].copy()

    for frame, cols in [(q, ["topic"]), (u, ["country", "device_type"]),
                         (e, ["variant", "country", "device_type"]), (a, ["activity_type"])]:
        for c in cols:
            frame[c] = frame[c].astype(str).str.strip().str.lower()

    u["signup_date"] = pd.to_datetime(u["signup_date"])
    q["created_date"] = pd.to_datetime(q["created_date"])
    a["activity_date"] = pd.to_datetime(a["activity_date"])
    e["assignment_date"] = pd.to_datetime(e["assignment_date"])

    # experiment.csv missingness is structural (0 questions asked -> undefined
    # average/rate), not random. Encode explicitly rather than guessing a value.
    e["has_asked_any_question"] = e["num_questions_asked"] > 0
    e["avg_word_count"] = e["avg_word_count"].fillna(0)
    # quality_answer_rate left as NaN: no answers exist to compute a rate from.

    return {"users": u, "questions": q, "user_activity": a, "experiment": e}


# --------------------------------------------------------------------------
# Leakage-safe derived variables
# --------------------------------------------------------------------------
def add_question_level_features(q: pd.DataFrame) -> pd.DataFrame:
    """Content/temporal features and the engineered engagement target."""
    q = q.copy()
    q["created_dow"] = q["created_date"].dt.day_name()
    q["created_is_weekend"] = q["created_date"].dt.dayofweek >= 5
    q["upvote_rate"] = (q["num_upvotes"] / q["num_views"]).replace([np.inf, -np.inf], np.nan)
    q["answer_rate"] = (q["num_answers"] / q["num_views"]).replace([np.inf, -np.inf], np.nan)

    # Composite engagement score: mean of min-max scaled views/answers/upvotes.
    scaled = q[["num_views", "num_answers", "num_upvotes"]].apply(
        lambda s: (s - s.min()) / (s.max() - s.min())
    )
    q["engagement_score"] = scaled.mean(axis=1)
    q["is_high_engagement"] = (
        q["engagement_score"] >= q["engagement_score"].quantile(0.75)
    ).astype(int)

    word_bins = [0, 15, 30, 60, 120, np.inf]
    word_labels = ["very_short", "short", "medium", "long", "very_long"]
    q["word_count_bucket"] = pd.cut(q["word_count"], bins=word_bins, labels=word_labels)
    return q


def add_asker_history_features(q: pd.DataFrame) -> pd.DataFrame:
    """
    Strictly-prior-in-time (leakage-safe) asker history: each question only
    "sees" the asker's own questions posted before it, never itself or later
    ones — computed via a per-user expanding window on chronologically
    sorted rows.
    """
    q_sorted = q.sort_values(["user_id", "created_date", "question_id"]).copy()
    q_sorted["asker_prior_questions_count"] = q_sorted.groupby("user_id").cumcount()
    cum_quality = (
        q_sorted.groupby("user_id")["got_quality_answer"].cumsum() - q_sorted["got_quality_answer"]
    )
    q_sorted["asker_prior_quality_rate"] = (
        cum_quality / q_sorted["asker_prior_questions_count"].replace(0, np.nan)
    )
    return q_sorted


def add_user_level_features(u: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Account-age features, bucketed for interpretability."""
    u = u.copy()
    u["account_age_at_ref_days"] = (reference_date - u["signup_date"]).dt.days
    u["signup_cohort_month"] = u["signup_date"].dt.to_period("M").astype(str)

    def age_bucket(days: float) -> str:
        if days < 90:
            return "new (<3mo)"
        if days < 365:
            return "established (3-12mo)"
        return "veteran (1yr+)"

    u["account_age_bucket"] = u["account_age_at_ref_days"].apply(age_bucket)
    return u


def aggregate_activity_to_question_level(a: pd.DataFrame) -> pd.DataFrame:
    """Per-question unique-actor counts by activity type (post-outcome; NOT a model feature)."""
    return (
        a.groupby(["question_id", "activity_type"])["user_id"]
        .nunique()
        .unstack(fill_value=0)
        .add_prefix("unique_")
        .add_suffix("_actors")
        .reset_index()
    )


# --------------------------------------------------------------------------
# Final assembly
# --------------------------------------------------------------------------
FEATURE_COLS = [
    "question_id", "user_id", "topic", "word_count", "word_count_bucket",
    "created_date", "created_dow", "created_is_weekend",
    "asker_country", "asker_device_type", "variant",
    "asker_account_age_at_question_days", "account_age_bucket",
    "asker_prior_questions_count", "asker_prior_quality_rate",
]
ENGAGEMENT_OUTCOME_COLS_BASE = [
    "num_views", "num_answers", "num_upvotes", "upvote_rate", "answer_rate",
    "engagement_score", "is_high_engagement",
]
TARGET_COL = ["got_quality_answer"]

# Data-leakage risk documentation (see README Limitations section):
#   HIGH  : num_answers, num_upvotes, num_views, unique_*_actors
#           -> post-outcome / same-day-concurrent signals for got_quality_answer
#   MODERATE: experiment.quality_answer_rate / avg_word_count
#           -> aggregated across a user's own questions; replaced by the
#              strictly-prior-in-time asker_prior_quality_rate instead
#   NONE  : word_count, topic, device_type, country, account age, asker prior
#           history (time-cut), day of week, experiment.variant


def build_modeling_dataset(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Runs the full Part-1 cleaning + feature-engineering pipeline and returns
    the final question-level modeling table (equivalent to data/modeling_dataset.csv).
    """
    clean = clean_raw(raw)
    q = add_question_level_features(clean["questions"])
    q_sorted = add_asker_history_features(q)

    reference_date = q["created_date"].max()
    u = add_user_level_features(clean["users"], reference_date)

    activity_agg = aggregate_activity_to_question_level(clean["user_activity"])

    modeling_df = (
        q_sorted
        .merge(u[["user_id", "country", "device_type", "signup_date",
                  "account_age_at_ref_days", "account_age_bucket"]],
               on="user_id", how="left")
        .merge(clean["experiment"][["user_id", "variant"]], on="user_id", how="left")
        .merge(activity_agg, on="question_id", how="left")
    )

    modeling_df["asker_account_age_at_question_days"] = (
        modeling_df["created_date"] - modeling_df["signup_date"]
    ).dt.days

    unique_actor_cols = [c for c in activity_agg.columns if c != "question_id"]
    modeling_df[unique_actor_cols] = modeling_df[unique_actor_cols].fillna(0)

    modeling_df = modeling_df.rename(
        columns={"country": "asker_country", "device_type": "asker_device_type"}
    )

    engagement_outcome_cols = ENGAGEMENT_OUTCOME_COLS_BASE + unique_actor_cols
    return modeling_df[FEATURE_COLS + engagement_outcome_cols + TARGET_COL]


if __name__ == "__main__":
    raw = load_raw("../data")  # expects users.csv, questions.csv, user_activity.csv, experiment.csv
    print("Referential integrity:", check_referential_integrity(raw))
    df = build_modeling_dataset(raw)
    print(f"Built modeling dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    df.to_csv("../data/modeling_dataset.csv", index=False)
