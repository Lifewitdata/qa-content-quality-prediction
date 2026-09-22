"""
feature_engineering.py
=======================
Text/NLP-derived features, target definition, and the final leakage-safe
feature set used for modeling. Direct refactor of the logic developed in
notebooks/02_eda_feature_engineering.ipynb (Part 2), operating on the
output of data_cleaning.build_modeling_dataset() (i.e. data/modeling_dataset.csv).

Scope note (see README): none of the four source extracts contain a raw
free-text field (no question title/body string) — only the pre-computed
`word_count` integer. Classic text features that require raw text
(character/sentence counts, average word length, readability, punctuation,
link/question-mark counts, TF-IDF/embeddings) are therefore NOT derivable
here and are not fabricated. Only what `word_count` legitimately supports
is engineered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Text-derived features (the only text signal this dataset supports)
# --------------------------------------------------------------------------
def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - log_word_count: log1p transform, corrects word_count's right skew
        (skew 1.83 -> ~0.04) for linear/distance-based models.
      - word_count_z_within_topic: a question's length expressed as a
        z-score relative to its own topic's mean/std, since "long" means
        something different in different topics. Derived only from
        word_count + topic (both features, not outcomes) -> no leakage risk.
    """
    df = df.copy()
    df["log_word_count"] = np.log1p(df["word_count"])

    topic_word_stats = df.groupby("topic")["word_count"].agg(["mean", "std"])
    df["word_count_z_within_topic"] = (
        (df["word_count"] - df["topic"].map(topic_word_stats["mean"]))
        / df["topic"].map(topic_word_stats["std"])
    )
    return df


# --------------------------------------------------------------------------
# Leakage-safe topic-history feature (mirrors asker_prior_* logic per-topic)
# --------------------------------------------------------------------------
def add_topic_prior_quality_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Each topic's cumulative quality rate computed only from questions
    posted strictly BEFORE the current one (expanding window, same
    discipline as asker_prior_quality_rate in Part 1). Cold-start rows
    (a topic's very first question) are filled with the dataset-wide rate.
    """
    df = df.copy()
    ts = df.sort_values(["topic", "created_date", "question_id"]).copy()
    grp = ts.groupby("topic")["got_quality_answer"]
    ts["topic_prior_count"] = grp.cumcount()
    cum_quality = grp.cumsum() - ts["got_quality_answer"]
    ts["topic_prior_quality_rate"] = cum_quality / ts["topic_prior_count"].replace(0, np.nan)

    overall_quality_rate = df["got_quality_answer"].mean()
    ts["topic_prior_quality_rate"] = ts["topic_prior_quality_rate"].fillna(overall_quality_rate)

    return df.merge(
        ts[["question_id", "topic_prior_count", "topic_prior_quality_rate"]],
        on="question_id", how="left",
    )


# --------------------------------------------------------------------------
# Target definition
# --------------------------------------------------------------------------
# Primary target: got_quality_answer -- a platform-recorded outcome flag
#   already present in questions.csv, not a threshold we chose. ~70/30 split.
# Secondary target: is_high_engagement -- engineered top-quartile (75th
#   percentile) cut of engagement_score (equal-weighted average of min-max
#   scaled views/answers/upvotes). Chosen over a fixed raw-count threshold
#   because it is scale-invariant and combines all three engagement signals.
#   Checked insensitive to the exact percentile used (70th-90th all tell the
#   same story). ~75/25 split.
TARGET_PRIMARY = "got_quality_answer"
TARGET_SECONDARY = "is_high_engagement"


def engagement_threshold_sensitivity(df: pd.DataFrame, quantiles=(0.70, 0.75, 0.80, 0.90)) -> pd.DataFrame:
    """Reproduces the Part-2 check that the target definition is not sensitive
    to the exact percentile cut chosen."""
    rows = []
    for q in quantiles:
        thresh = df["engagement_score"].quantile(q)
        pct_positive = (df["engagement_score"] >= thresh).mean()
        rows.append({"quantile": q, "engagement_score_threshold": thresh, "pct_flagged_positive": pct_positive})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Final leakage-safe feature groups (Part 2)
# --------------------------------------------------------------------------
BEHAVIORAL_FEATURES = [
    "asker_prior_questions_count", "asker_prior_quality_rate",
    "topic_prior_count", "topic_prior_quality_rate",
]
CONTENT_FEATURES = [
    "word_count", "log_word_count", "word_count_bucket", "word_count_z_within_topic",
]
USER_FEATURES = [
    "asker_country", "asker_device_type",
    "asker_account_age_at_question_days", "account_age_bucket",
]
CATEGORY_FEATURES = ["topic"]
TEMPORAL_FEATURES = ["created_dow", "created_is_weekend"]
EXPERIMENT_FEATURES = ["variant"]

ALL_FEATURES = sorted(set(
    BEHAVIORAL_FEATURES + CONTENT_FEATURES + USER_FEATURES
    + CATEGORY_FEATURES + TEMPORAL_FEATURES + EXPERIMENT_FEATURES
))

# Never used as model inputs for either target -- outcomes / same-day
# concurrent signals (see data_cleaning.py leakage documentation).
LEAKAGE_EXCLUDED = [
    "num_views", "num_answers", "num_upvotes", "upvote_rate", "answer_rate",
    "engagement_score", "is_high_engagement",
    "unique_answer_actors", "unique_ask_actors", "unique_upvote_actors", "unique_view_actors",
]


def build_modeling_dataset_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs the full Part-2 pipeline on top of Part 1's modeling_dataset.csv:
    text features -> topic-prior feature -> final column assembly.
    Equivalent to data/modeling_dataset_v2.csv.
    """
    df = add_text_features(df)
    df = add_topic_prior_quality_rate(df)

    final_columns = (
        ["question_id", "user_id", "created_date"]
        + ALL_FEATURES + LEAKAGE_EXCLUDED + [TARGET_PRIMARY, TARGET_SECONDARY]
    )
    seen = set()
    final_columns = [c for c in final_columns if not (c in seen or seen.add(c))]
    return df[final_columns].copy()


if __name__ == "__main__":
    df = pd.read_csv("../data/modeling_dataset.csv", parse_dates=["created_date"])
    df_v2 = build_modeling_dataset_v2(df)
    print(f"Built Part-2 modeling dataset: {df_v2.shape[0]:,} rows x {df_v2.shape[1]} columns")
    print(f"Leakage-safe features ({len(ALL_FEATURES)}): {ALL_FEATURES}")
    df_v2.to_csv("../data/modeling_dataset_v2.csv", index=False)
