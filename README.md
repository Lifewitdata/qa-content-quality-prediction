# Content Quality & Engagement Prediction — Q&A Platform

A four-stage, Python-only data science project: raw data → EDA & cleaning →
feature engineering & NLP → machine learning → model interpretation & error
analysis → **product application**. Built to demonstrate the kind of applied
data-science work relevant to a Data Scientist role at a Q&A / content
platform: defensible data cleaning, leakage-aware feature
engineering, imbalance-aware modeling, honest error analysis, and product
recommendations that follow directly from the evidence — not from wishful
thinking about the model.

> **No fabricated numbers.** Every metric, coefficient, percentage, and
> table in this README is either read directly from executed notebook
> output or recomputed from `data/modeling_dataset_v2.csv` in this repo.
> Where the data can't support a claim (e.g. deep NLP, causal effects), that
> limitation is stated explicitly rather than glossed over.

**Results at a glance**

| | |
|---|---|
| Rows / grain | 37,719 questions, one row per `question_id` |
| Primary (recorded) target | `got_quality_answer` — 69.7% positive |
| Modeled target | `is_high_engagement` (top-quartile engagement) — 25.0% positive |
| Best model | Logistic Regression |
| Test PR-AUC | **0.395** vs. base-rate floor **0.255** |
| Test ROC-AUC | **0.665** |
| Dominant feature | `log_word_count` (question length) — ~13x every other feature |
| Model strength | 92.7% accurate on very-short questions (reliable filter) |
| Model weakness | 44–55% accurate on medium/long questions (near coin-flip) |

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Analytical Questions](#3-analytical-questions)
4. [Dataset](#4-dataset)
5. [Data Cleaning](#5-data-cleaning)
6. [Feature Engineering](#6-feature-engineering)
7. [NLP Features](#7-nlp-features)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Target Definition](#9-target-definition)
10. [Machine Learning Methodology](#10-machine-learning-methodology)
11. [Model Comparison](#11-model-comparison)
12. [Model Evaluation](#12-model-evaluation)
13. [Feature Importance](#13-feature-importance)
14. [Error Analysis](#14-error-analysis)
15. [Key Findings & Business Analysis](#15-key-findings--business-analysis)
16. [Product Applications](#16-product-applications)
17. [Recommendations](#17-recommendations)
18. [Limitations](#18-limitations)
19. [Tech Stack](#19-tech-stack)
20. [Project Structure](#20-project-structure)

---

## 1. Project Overview

This project works through four raw CSV extracts from a Q&A
platform (`users`, `questions`, `user_activity`, `experiment`) end-to-end:

`Data → Product Question → EDA → Feature Engineering → NLP → Machine Learning → Model Evaluation → Error Analysis → Product Decision`

It is organized as four notebooks plus a mirrored, importable `src/`
package, so every step is both interactively explorable and reproducible
from the command line.

- **Part 1 — Data Profiling** (`01_data_profiling.ipynb`): inspection,
  referential-integrity checks, missing values, duplicates, outliers,
  leakage audit, cleaning, and construction of the first modeling table.
- **Part 2 — EDA & Feature Engineering** (`02_eda_feature_engineering.ipynb`):
  deeper exploratory analysis, the only NLP features this dataset legitimately
  supports, target definition, and the final leakage-safe feature set.
- **Part 3a — Modeling** (`03_modeling.ipynb`): chronological
  train/validation/test split, four classifiers, imbalance-aware comparison,
  and model selection.
- **Part 3b — Model Interpretation & Error Analysis**
  (`04_model_interpretation.ipynb`): coefficients, permutation importance,
  SHAP, and a structured error analysis.
- **This README**: translates all of the above into product applications,
  a business-analysis Q&A, Finding→Evidence→Action recommendations, and an
  honest limitations section.

No machine learning is trained until Part 3; Parts 1–2 exist to produce a
clean, well-understood, leakage-checked dataset first.

## 2. Business Problem

A Q&A platform's long-term health depends on two closely related outcomes:
whether questions **get a quality answer** (the core content-quality
promise to users) and whether content **generates engagement** (views,
answers, upvotes — the core growth/retention loop). Today, whether either
of these happens is only known *after the fact*. If the platform could get
even a moderately reliable, pre-outcome read on which new questions are
likely to underperform, several teams — content operations, growth,
trust & safety, and product — could act earlier and more precisely instead
of reacting after engagement has already been decided.

This project asks a scoped, honest version of that question: **using only
information available at or before publish time, how well can we predict
content quality/engagement, what actually drives the prediction, and where
exactly should — and shouldn't — a product team trust it?**

## 3. Analytical Questions

1. What question- and user-level characteristics are associated with
   getting a quality answer / becoming high-engagement content?
2. Can either outcome be predicted ahead of time using only leakage-safe
   (available-before-the-outcome) signals?
3. Which features matter most, and how robust is that ranking across
   different interpretation methods?
4. Do quality/engagement patterns differ meaningfully by topic, country,
   device, or asker account age — enough to justify segment-specific
   product strategies?
5. Does the platform's existing A/B test (`variant`) show a measurable
   difference in quality/engagement?
6. Where does a trained model succeed and fail, and what does that error
   pattern imply about how (and how *not*) to use it in the product?

## 4. Dataset

Four raw CSV extracts (synthetic, platform-style data):

| File | Grain | Rows | Key columns |
|---|---|---|---|
| `users.csv` | 1 row / user | 12,000 | `user_id`, `signup_date`, `country`, `device_type` |
| `questions.csv` | 1 row / question | 37,719 | `question_id`, `user_id`, `topic`, `word_count`, `num_views`, `num_answers`, `num_upvotes`, `got_quality_answer`, `created_date` |
| `user_activity.csv` | 1 row / event | 215,370 | `activity_id`, `user_id`, `question_id`, `activity_type` (view/upvote/answer/ask), `activity_date` |
| `experiment.csv` | 1 row / user | 12,000 | `user_id`, `variant` (A/B assignment), per-user aggregates |

Inferred schema:
```
users(user_id) --------------------< questions(user_id)
users(user_id) --------------------< user_activity(user_id)
questions(question_id) ------------< user_activity(question_id)
users(user_id) ---------------------- experiment(user_id)         [1:1]
```

Observation window: **2026-06-01 to 2026-07-30** (~9 weeks), inferred from
the chronological train/val/test date ranges used in modeling.

**What's included in `data/` here:** the two *processed* modeling tables
produced by Parts 1 and 2 —

- `modeling_dataset.csv` (37,719 rows × 27 cols) — Part 1 output
- `modeling_dataset_v2.csv` (37,719 rows × 31 cols) — Part 2 output, adds
  NLP-derived and topic-history features
- `data_quality_summary.md` — the Part-1 written data-quality summary

The four **raw** extracts are not redistributed in this repo (see
`notebooks/01_data_profiling.ipynb` for the loading/cleaning code that
would run against them). Notebooks 02–04 and all of `src/` run standalone
against the processed tables that *are* included.

## 5. Data Cleaning

All checks below were run against the raw extracts in Part 1
(`src/data_cleaning.py`, `notebooks/01_data_profiling.ipynb`):

- **Referential integrity:** 0 orphaned foreign keys in any direction
  across all four tables; `users` ↔ `experiment` confirmed 1:1.
- **Duplicates:** 0 duplicate IDs and 0 full-row duplicates in any file.
  A small number of `user_activity` rows share the same
  `(user_id, question_id, activity_type)` — kept as legitimate repeat
  behavior (e.g. viewing a question twice), not treated as errors.
- **Missing values:** confined entirely to `experiment.csv`
  (`avg_word_count`, `quality_answer_rate`), and are **structural**: among
  users with `num_questions_asked == 0`, 100% have these fields missing
  (an undefined average/rate over zero questions), not random.
- **Category consistency:** no casing/whitespace variants found in any
  categorical column; `users.csv` and `experiment.csv` category values
  agree per-user with 0 mismatches.
- **Invalid values:** 0 rows with negative counts, out-of-range rates, or
  `got_quality_answer == 1` while `num_answers == 0` (logically impossible).
- **Dates:** fully internally consistent — 0 questions predate their
  asker's signup, 0 activity predates its question. **Activity is logged
  at day-level granularity only, and always falls on the question's
  creation day** — this single fact rules out any genuine "early
  engagement" (e.g. views in the first hour) feature anywhere in this
  project, and is referenced repeatedly below.
- **Cross-source consistency:** `questions.csv`'s own engagement counters
  (`num_views`/`num_answers`/`num_upvotes`) do **not** match an independent
  aggregation of `user_activity.csv` for the same questions —
  `user_activity.csv` is an independent, partial event log, not the source
  of the summary counters. Both are kept as separate signal sources.
- **Outliers:** IQR-method outliers exist in `num_views`/`num_upvotes`
  (~445 high-upvote questions) with no negative or impossible values —
  read as genuine viral questions, not data errors, and **kept**.

## 6. Feature Engineering

All features below are asserted leakage-safe with an explicit audit (see
[Limitations](#18-limitations) for one residual caveat on that audit).

- **Behavioral (strictly prior-in-time):** `asker_prior_questions_count`,
  `asker_prior_quality_rate` — computed via a per-user **expanding window**
  over chronologically sorted rows, so a question only "sees" that asker's
  *earlier* questions, never itself or later ones. `topic_prior_count`,
  `topic_prior_quality_rate` — the same discipline applied per-topic
  (cold-start topics filled with the dataset-wide rate).
- **Content:** `word_count`, `word_count_bucket` (`very_short` / `short` /
  `medium` / `long` / `very_long`, bin edges at 15/30/60/120 words).
- **User:** `asker_country`, `asker_device_type`,
  `asker_account_age_at_question_days` (account age **at the time of that
  specific question**, not a fixed reference date), `account_age_bucket`
  (`new <3mo` / `established 3-12mo` / `veteran 1yr+`).
- **Category:** `topic` (10 topics).
- **Temporal:** `created_dow`, `created_is_weekend`.
- **Experiment:** `variant` (A/B assignment — randomized and pre-assigned,
  so safe as a predictive feature).
- **Engineered engagement composite:** `engagement_score` = mean of
  min-max-scaled `num_views`, `num_answers`, `num_upvotes` (an **outcome**,
  never a feature; source of the `is_high_engagement` target — see
  [§9](#9-target-definition)).

Explicitly **excluded** from the model's feature set — see the leakage
audit in `src/data_cleaning.py` and `src/feature_engineering.py`:
`num_views`, `num_answers`, `num_upvotes`, `upvote_rate`, `answer_rate`,
`engagement_score`, `is_high_engagement`, and all `user_activity`-derived
unique-actor counts. `num_answers` alone has **zero** counter-examples
where `got_quality_answer == 1` and `num_answers == 0` — i.e. it is a
near-perfect proxy for that target by construction.

## 7. NLP Features

**Scope note, stated plainly rather than worked around:** none of the four
source extracts contain a raw free-text field (no question title or body
string) — only the pre-computed `word_count` integer. This rules out most
"classic" NLP features:

| Feature | Derivable here? | Why |
|---|---|---|
| Word count | ✅ Yes | Already provided (`word_count`) |
| Character count | ❌ No | No raw string to count characters from |
| Sentence count | ❌ No | No raw string to split into sentences |
| Average word length | ❌ No | Requires characters ÷ words |
| Readability score (e.g. Flesch) | ❌ No | Requires sentence + syllable structure |
| Punctuation features | ❌ No | Requires raw string |
| Number of links | ❌ No | Requires raw string |
| Question-mark count | ❌ No | Requires raw string |
| TF-IDF / embeddings | ❌ No | Requires a text corpus, not one summary number |

What **was** built, from what `word_count` legitimately supports:

- **`log_word_count`** — `log1p(word_count)`; corrects the raw feature's
  right skew (**1.83 → 0.04**), helping linear/distance-based models.
- **`word_count_z_within_topic`** — a question's length expressed as a
  z-score relative to its own topic's mean/std (40 words is short for
  `career`, long for `sports`); derived only from `word_count` + `topic`
  (both features, not outcomes), so it carries no leakage risk.
- `word_count_bucket` (from Part 1) also serves as an interpretable,
  non-linear-effect-friendly categorical view of length.

This gap is documented, with a checklist for what to build once raw text
becomes available, in `notebooks/01_data_profiling.ipynb` (Text-Readiness
section) and `notebooks/02_eda_feature_engineering.ipynb` (§2).

## 8. Exploratory Data Analysis

(Full analysis in `notebooks/02_eda_feature_engineering.ipynb`; figures in
`outputs/figures/p2_*.png`.)

- **Engagement distributions:** views and upvotes are right-skewed with a
  long tail of high-performing questions (consistent with the outlier
  check in §5 — plausible viral content, not errors). ~44% of questions
  get zero answers; most that do get answered receive exactly one.
- **Time:** question volume and quality/engagement rates are essentially
  **flat over the ~9-week window** — no meaningful trend, so calendar time
  itself is not a strong predictive feature.
- **Topic:** real but **modest** differences. Quality rate ranges from
  **68.4% (sports) to 71.4% (travel)** — see `outputs/tables/topic_summary.csv`
  for the full breakdown (travel 71.4%, relationships 71.0%, politics
  70.3%, science 70.0%, education 69.8%, technology 69.3%, finance 69.1%,
  health 68.9%, career 68.5%, sports 68.4%).
- **Question length — the strongest single driver found:** quality rate
  climbs **monotonically** from **55.8%** (very_short) to **80.8%**
  (very_long) across length buckets — a 25-point spread, far larger than
  any topic/device/country effect (`outputs/tables/quality_by_word_count_bucket.csv`).
- **The A/B `variant` — a large, consistent effect:** `treatment` users
  post longer questions on average (**48.1 vs 33.4 words**) *and* have a
  meaningfully higher quality rate (**71.6% vs 67.7%**) and higher average
  upvotes (**6.71 vs 5.91**) (`outputs/tables/variant_comparison.csv`).
  Worth a dedicated causal/uplift read, not just a predictive one — see
  [§18](#18-limitations) for why this project stops short of a causal claim.
- **Device, country, account age:** only marginal (<2-point) differences —
  weak signals at best (device 69.4–69.7%; country 68.3–70.7%; account age
  68.3–69.8%).
- **Correlations with content length:** `word_count` correlates positively
  with every engagement signal (views r=0.33, `engagement_score` r=0.27,
  `got_quality_answer` r=0.115–0.12, upvotes r=0.18) — none strong alone,
  but combined with the monotonic bucket pattern, length is clearly the
  most consistent content-level driver in this dataset.

## 9. Target Definition

**Primary (recorded) target — `got_quality_answer`:** a platform-recorded
binary outcome already present in `questions.csv`, not a threshold we
chose — verified internally consistent (a question can only be flagged
`1` if it received ≥1 answer). Split: **69.7% positive / 30.3% negative**
(~2.3:1).

**Modeled target — `is_high_engagement`:** no ready-made "high engagement"
flag exists, so one was engineered as the **top quartile (75th percentile)
of `engagement_score`** (the equal-weighted, min-max-scaled average of
views/answers/upvotes from §6). A percentile cut was used instead of a
fixed raw-count threshold because it is scale-invariant and combines all
three engagement signals rather than relying on one. The 75th-percentile
cut was checked for sensitivity — 70th/75th/80th/90th percentile cuts all
tell the same story (`outputs/tables` reproduces this check via
`src/feature_engineering.engagement_threshold_sensitivity`). Split:
**75.0% negative / 25.0% positive** (~3:1).

**Why Part 3 models `is_high_engagement` rather than `got_quality_answer`:**
`got_quality_answer`'s strongest possible predictor, `num_answers`, is
excluded as direct leakage by construction (§6), and `is_high_engagement`
is the more product-relevant, blended "did this content perform well"
framing that ties quality and engagement together in one target — which
matches this project's stated goal.

Both targets are **moderate**, not severe, imbalances — addressed with
`class_weight="balanced"` / `scale_pos_weight`, PR-AUC/ROC-AUC as the
primary metrics, and minority-class F1/precision/recall, rather than
synthetic resampling (SMOTE etc., not needed at this imbalance level).

## 10. Machine Learning Methodology

**Split — chronological, not random** (`src/modeling.chronological_split`):
train on the past, validate/test on data the model has never seen
chronologically — the realistic simulation of scoring newly-posted
questions. A random split would leak information, since features like
`word_count_z_within_topic` are computed from dataset-wide statistics.

| Split | Rows | Date range | Positive rate |
|---|---|---|---|
| Train | 26,403 | 2026-06-01 → 2026-07-12 | 25.0% |
| Validation | 5,658 | 2026-07-12 → 2026-07-21 | 24.6% |
| Test | 5,658 | 2026-07-21 → 2026-07-30 | 25.5% |

Class balance is preserved across all three splits purely as a byproduct
of the target being flat over time (§8) — no manual stratification needed.

**Preprocessing** (`src/modeling.build_preprocessors`): numeric features
median-imputed (fit on train only); categorical features one-hot encoded
with `handle_unknown="ignore"`. Logistic Regression additionally gets a
`StandardScaler` (tree models are scale-invariant). `asker_prior_quality_rate`
is null for first-time askers — a real, informative null (36.1% of train
rows, 9.7% of validation, 6.8% of test, reflecting the growing pool of
returning askers over the window) — median-imputed here; tree models could
alternatively handle it natively.

**Baseline:** a majority-class `DummyClassifier` — **75.4% accuracy** with
**0 recall** on the class that matters, and **PR-AUC = 0.246** (the
positive-class base rate — the real floor for PR-AUC on this data).
This is exactly why accuracy is excluded from model selection below.

**Models trained** (`src/modeling.build_models`), each chosen for a
specific reason:
1. **Logistic Regression** (`class_weight="balanced"`) — fast, fully
   interpretable; coefficients translate directly to business language.
2. **Random Forest** (`class_weight="balanced_subsample"`) — captures
   non-linear effects/interactions without manual feature crosses.
3. **HistGradientBoostingClassifier** (`class_weight="balanced"`) —
   usually the strongest tabular performer; native missing-value handling.
4. **XGBoost** (`scale_pos_weight`) — included as the requested additional
   strong model; this dataset (mixed numeric/categorical, moderate size,
   real but non-trivial signal) is exactly the setting where boosting
   variants tend to add value over a single Random Forest.

All imbalance-sensitive hyperparameters use the **train-set** class ratio
only (3.01:1) — never validation/test information. Hyperparameters reflect
a light, validation-PR-AUC-guided manual search, not an exhaustive grid —
appropriate for a defensible model comparison, not squeezing out the last
0.5% of performance.

**Evaluation metrics:** PR-AUC first (threshold-independent, reflects
minority-class performance under imbalance), ROC-AUC second, then
F1/precision/recall at the default 0.5 threshold. **Accuracy is
intentionally excluded from model selection** (§ baseline above).

## 11. Model Comparison

Full numbers in `outputs/tables/model_comparison.csv`; reproduced by
`notebooks/03_modeling.ipynb` and `src/evaluation.evaluate_all`.

**Validation set:**

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Dummy (majority) | 0.2464 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** | **0.3609** | **0.6452** | 0.4356 | 0.3327 | 0.6306 |
| Random Forest | 0.3608 | 0.6437 | 0.4204 | 0.3397 | 0.5516 |
| Gradient Boosting (HGB) | 0.3587 | 0.6445 | 0.4305 | 0.3368 | 0.5961 |
| XGBoost | 0.3539 | 0.6409 | 0.4321 | 0.3324 | 0.6169 |

**Test set** (touched once, after selection on validation):

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Dummy (majority) | 0.2552 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** | **0.3954** | **0.6645** | 0.4547 | 0.3490 | 0.6524 |
| Random Forest | 0.3894 | 0.6603 | 0.4427 | 0.3589 | 0.5776 |
| Gradient Boosting (HGB) | 0.3908 | 0.6603 | 0.4457 | 0.3477 | 0.6205 |
| XGBoost | 0.3769 | 0.6507 | 0.4440 | 0.3380 | 0.6468 |

All four trained models beat the dummy baseline by a wide, real margin.
The four cluster tightly with each other (within ~0.02 PR-AUC) — this
dataset's signal is dominated by one feature (§13), which caps how much
tree-based non-linearity/interaction-modeling can add over a
well-specified linear model.

**Selection rule** (decided *before* looking at test numbers): rank by
validation PR-AUC first, use ROC-AUC and minority-class F1 as tie-breakers,
and prefer the simpler model when performance is statistically
indistinguishable. **Logistic Regression wins validation PR-AUC and
ROC-AUC outright**, and — being the most interpretable and cheapest to
deploy/monitor — is selected as the primary model for interpretation and
error analysis. The tree-based models are kept as viable, near-equivalent
alternatives, not discarded.

## 12. Model Evaluation

Final, single-touch evaluation of Logistic Regression on the held-out test
set (`outputs/tables/test_set_error_breakdown.csv`):

- **PR-AUC 0.395** vs. base-rate floor **0.255** — a real, moderate lift
  (~55% relative improvement over the floor), but leaves substantial
  variance unexplained.
- **ROC-AUC 0.665**, **Recall 0.652**, **Precision 0.349**.
- **Confusion matrix (test, threshold 0.5):**

  | | Predicted: Not high-engagement | Predicted: High-engagement |
  |---|---|---|
  | **Actual: Not high-engagement** | TN = 2,457 | FP = 1,757 |
  | **Actual: High-engagement** | FN = 502 | TP = 942 |

**Reading:** at the default threshold the model catches about **65% of
genuinely high-engagement content** (recall), but roughly **2 in 3
questions it flags as "high engagement" won't actually be** (precision
0.349). This is a real, usable signal for triage and soft ranking — not
an autonomous decision-maker. The right operating threshold for any real
product use should be chosen based on the actual cost of a false positive
vs. a false negative in that specific use case (§16–17), not defaulted to
0.5.

## 13. Feature Importance

Three complementary methods (`notebooks/04_model_interpretation.ipynb`),
deliberately in increasing order of robustness to multicollinearity:

1. **Standardized coefficients** — fastest, but `word_count` and
   `log_word_count` are highly correlated (same underlying quantity), so
   the model can split their combined effect in a way that makes raw
   `word_count`'s coefficient look weakly *negative* even though longer
   questions clearly perform better. A textbook multicollinearity
   artifact, not a "shorter is better" finding — hence methods 2–3.
2. **Permutation importance** (validation set, PR-AUC drop;
   `outputs/tables/permutation_importance.csv`) — shuffles one *raw*
   feature at a time; not affected by the coefficient-splitting issue:

   | Feature | Mean PR-AUC drop | Std |
   |---|---|---|
   | **`log_word_count`** | **0.1295** | 0.0053 |
   | `asker_prior_quality_rate` | 0.0101 | 0.0047 |
   | `asker_device_type` | 0.0027 | 0.0007 |
   | `word_count` | 0.0025 | 0.0021 |
   | `word_count_z_within_topic` | 0.0022 | 0.0020 |
   | `topic` | 0.0019 | 0.0014 |
   | `asker_country` | 0.0014 | 0.0013 |
   | `asker_prior_questions_count` | 0.0009 | 0.0005 |
   | `variant` | 0.0005 | 0.0007 |
   | all remaining features | ≤ 0.0001 (noise-level) | — |

3. **SHAP** (additive, per-prediction; `outputs/figures/p3_07_shap_summary.png`)
   — confirms both prior methods: `log_word_count` has by far the widest
   spread of SHAP values; `asker_prior_quality_rate` and
   `word_count_z_within_topic` contribute modestly; everything else
   contributes only small, per-observation nudges.

**Business summary:** the highest-leverage lever the platform has is
**encouraging longer, more detailed questions** — by roughly an order of
magnitude over every other feature. `asker_prior_quality_rate` (a
reputation/track-record effect) is a modest secondary lever. Topic
(science/relationships/technology positive; career/sports negative),
country (Brazil/US/Philippines positive; Germany/Canada negative), device
type, day-of-week, account age, and the A/B `variant` are all near-zero
standalone predictors in the model. This is worth flagging explicitly:
**`variant` shows a real *descriptive* difference (§8) but contributes
almost nothing once `word_count` is in the model** — consistent with
`variant`'s effect being *mediated through* question length rather than
acting as an independent channel. That is a mediation hypothesis, not a
confirmed causal decomposition (§18).

## 14. Error Analysis

Full detail in `notebooks/04_model_interpretation.ipynb`, §9.

**Accuracy by content length** (`outputs/tables/accuracy_by_length_bucket.csv`)
— the single most important error-analysis finding:

| `word_count_bucket` | Test accuracy |
|---|---|
| very_short | **92.7%** |
| short | 82.9% |
| medium | 45.3% |
| long | 44.0% |
| very_long | 54.9% |

The model is excellent at the *extremes* and drops to barely-better-than-
coin-flip in the middle/long range. **Length reliably rules out low
performers; among already-long questions, length stops being a useful
discriminator, and the model has no other strong signal to fall back on.**
In other words: **length is a good filter, not a good ranker.**

**False positives vs. false negatives**
(`outputs/tables/test_set_error_breakdown.csv`):

- **False positives** (predicted high engagement, wasn't): mean
  `word_count` **56.5** — nearly as long as true positives (64.1). Long,
  detailed questions that *should* have done well by the model's one
  strong heuristic but didn't — plausibly well-written but niche or
  poorly-timed content the model can't distinguish from a genuine winner.
- **False negatives** (missed genuine high-engagement content): mean
  `word_count` **27.2** — short questions that did well anyway, e.g. via a
  compelling topic, an early upvote cascade, or an engaged-audience asker
  this feature set only partially captures via `asker_prior_quality_rate`.

**By topic:** error rates are fairly close together (FP rates 36–48%, FN
rates 30–42%) — no topic is badly broken, consistent with the modest topic
effects in §8. `finance` and `education` have the highest false-negative
rates; `science`, `finance`, and `travel` have the highest false-positive
rates.

**Concrete examples** (test set): the highest-confidence false positive is
a 131-word `science` question from an asker with a perfect prior quality
rate (predicted probability 0.745) that still wasn't high-engagement. The
highest-confidence false negative is an 11-word `technology` question from
an asker with a 0% prior quality rate (predicted probability 0.160) that
*was* high-engagement anyway — the kind of case no feature here can see coming.

**Data-quality contributors to these errors:**
- **No raw question text** — the model sees `word_count`, not *what was
  written*; two equally-long questions can differ wildly in clarity,
  specificity, or topical relevance. Very likely the single biggest
  ceiling on performance (§7).
- **No comments data** — a plausible early, pre-outcome engagement signal
  that isn't available at all.
- **Day-level-only activity timestamps** (§5) — no way to build genuine
  "early engagement" features for a still-unanswered question.
- **`engagement_score`'s equal weighting is a judgment call** (§9/§18) — a
  different weighting could shift which questions are labeled positive.
- The near-total dominance of one feature and flat segment effects are
  consistent with either a genuinely simple underlying process or a
  dataset with less naturalistic complexity than a live production
  platform — results here are **directionally instructive, not a
  performance guarantee** on real platform data.

## 15. Key Findings & Business Analysis

### Headline findings
- Question **length is the strongest, most consistent driver** of both
  quality and engagement in this dataset — by a wide margin over every
  other feature, in both the descriptive EDA and the trained model.
- The trained classifier provides a **real but moderate** lift over a
  naive baseline (test PR-AUC 0.395 vs. 0.255 floor) and is **strong at
  the extremes, weak in the high-volume middle** — a filter, not a ranker.
- Everything else in the current feature set (topic, country, device,
  day-of-week, account age, asker history, the A/B variant) is a **minor
  refinement**, not an independent strategy, on top of the length effect.

### Business-analysis Q&A

**1. What characteristics are associated with high-performing content?**
Primarily **length/detail**: quality rate rises monotonically from 55.8%
(very-short) to 80.8% (very-long) across length buckets. Secondarily, a
modest **asker track record** effect (`asker_prior_quality_rate`), a small
**topic** spread (travel/relationships lead, career/sports trail by ~3
points), a small **country** spread (GB/AU/PH lead, DE trails by ~2
points), and the **A/B `treatment`** group (longer questions, higher
quality rate, more upvotes than control) — though the model's own
interpretation (§13) suggests this last effect runs largely *through*
length rather than independently of it.

**2. Which features are most predictive?**
`log_word_count` dominates, by roughly an order of magnitude, in
permutation importance and SHAP alike. `asker_prior_quality_rate` is the
only other feature with a non-negligible standalone effect. Every other
feature — topic, country, device, day-of-week, account age, and the A/B
variant — contributes only small individual signal (§13 table).

**3. Which content categories perform differently?**
Topic differences are **real but modest**: a ~3-point quality-rate spread
(68.4% sports → 71.4% travel) and a similarly modest spread in
high-engagement rate. No topic is a dramatic outlier in either
performance or error rate (§14) — this is not a "some categories are
fundamentally broken" story.

**4. Where does the model perform well?**
On **very-short and short content** — 92.7% and 82.9% test accuracy
respectively — where length alone is a strong, reliable signal that
content is unlikely to perform well. The model also clears the naive
baseline by a real margin overall (PR-AUC +55% relative to the base rate).

**5. Where does it perform poorly?**
On **medium, long, and very-long content** — 44–55% test accuracy, close
to a coin flip — where length has already "used up" its discriminating
power and no other feature in the set is strong enough to take over.
Precision at the default threshold (0.349) is also modest platform-wide,
meaning a majority of "flagged as high-engagement" predictions won't pan out.

**6. What could product teams do with these insights?**
Use the model as a **confident low-effort-content filter and a soft
triage signal**, never as an autonomous ranker or gatekeeper for the bulk
of content (which falls in its weak middle zone). Treat length as the
platform's primary, low-risk quality lever (compose-time nudges). Treat
the A/B `variant` finding as a strong hypothesis for a dedicated causal
follow-up, not a confirmed causal win. Deprioritize topic/country/device-
specific interventions given how modest those effects are. See §16–17 for
the full, evidence-linked product application and recommendations.

**7. What additional data would improve the model?**
**Raw question (and ideally answer) text** — enabling real NLP (TF-IDF/
embeddings, readability, clarity/specificity) where §7's derivability
table shows nothing is currently possible. **Comments data** (not present
in any extract) as an early engagement signal. **Sub-day/timestamp-level
activity logs** (§5) to build genuine "early signal" features instead of
same-day-only aggregates. **Answer-level quality features** (not just
presence/absence of a quality answer). **Asker network/follower data**,
to test the "engaged audience" hypothesis raised in the false-negative
error analysis (§14). A **documented, possibly multiple, engagement-target
definition(s)** to test how sensitive findings are to the equal-weighting
judgment call in `engagement_score` (§9, §18).

## 16. Product Applications

Predictions from this model are a **candidate input signal**, not a
decision-maker. Every application below is scoped to where the error
analysis (§14) shows the model is actually reliable, and every one keeps a
human or an existing product process in the loop for anything the model is
not confident about.

- **Content discovery.** The model's high accuracy on very-short/short
  content (§14) makes it a reasonable *de-prioritization* signal for
  "promising new questions" surfaces — i.e., confidently filtering out
  content unlikely to perform, rather than confidently promoting content
  in the (much larger, much less reliable) medium/long range. Any
  "trending" or "promising" surfacing should combine this signal with
  real, accruing engagement data as it becomes available, not rely on the
  pre-publish prediction alone.
- **Content ranking.** Given 0.349 precision at the default threshold and
  near-coin-flip accuracy in the medium/long range (the bulk of content by
  volume), this model **should not be a primary ranking signal**. It could
  contribute as one weak input among several in a larger ranking model,
  particularly for very fresh content where no real engagement signal
  exists yet — but it should be weighted low and monitored for drift.
- **Quality review.** The most defensible "prioritization" use case: route
  the human review queue toward the medium/long/very-long content where
  the model is least confident (and where human judgment adds the most
  marginal value), and let very-short, confidently-low-scoring content
  consume less reviewer time. Reviewers retain final judgment throughout.
- **Answer recommendations.** Out of scope for this model as built — there
  is no answer-level model here, only a whole-question outcome. A natural,
  separate extension: surface predicted-high-engagement-but-still-
  unanswered questions to subject-matter/expert answerers first, using
  this model's score purely as one prioritization input, not a guarantee.
- **Feed personalization.** Length/topic signals could be blended into a
  larger feed-ranking model as a weak prior, but given how modest topic/
  segment effects are (§8, §13), this signal should never dominate
  personalization — over-weighting it risks narrowing feed diversity for
  a small, unproven benefit.
- **Moderation prioritization.** This model does **not** detect harmful,
  unsafe, or policy-violating content — that is entirely out of scope. The
  general triage *pattern* demonstrated here (a soft probability score
  routing content to a human review queue, with confident low-risk cases
  automated and everything else escalated) is directly transferable to a
  properly-scoped moderation-triage system, but would need its own model
  trained on actual moderation labels. Predicted-quality scores should
  never be a sole trigger for removal, suppression, or other high-impact
  moderation actions.
- **Creator/content feedback.** The **strongest, most product-ready**
  application: a lightweight, optional compose-time nudge for very-short
  questions ("similar short questions rarely get a detailed answer —
  consider adding more context"), directly supported by the strong,
  monotonic length-quality relationship (§8) and the model's high accuracy
  specifically in that region (§14). This keeps the decision with the
  person writing the question, rather than the platform acting on their
  behalf.

**On human/product oversight (applies to all of the above):** this model
should **never autonomously demote, hide, remove, or promote** content, and
should never be the sole basis for an irreversible action. Given that its
accuracy falls to near-chance for the majority-volume medium/long content
band, any automated action driven by its score should be gated to the
narrow regions where it is demonstrably reliable (very-short/short
content), with everything else routed to human review or left to organic,
real engagement signals to resolve. Predictions should be monitored for
drift and re-validated periodically, not treated as a fixed, permanent
truth about content quality.

## 17. Recommendations

Each recommendation follows directly from a finding already established
above — no new claims are introduced here.

---

**Recommendation 1 — Nudge very-short questions at compose time.**
- **Finding:** Question length is by far the strongest predictor of
  quality/engagement, and the model is highly accurate specifically in the
  very-short/short region.
- **Evidence:** Quality rate rises monotonically 55.8% → 80.8% across
  length buckets (§8); `log_word_count` permutation importance (0.1295) is
  ~13x the next-highest feature (§13); model test accuracy is 92.7%/82.9%
  on very-short/short content (§14).
- **Product implication:** Encouraging more detail on short questions is
  the platform's single highest-leverage, lowest-risk content-quality lever.
- **Recommended action:** Add an optional, dismissible compose-time prompt
  for very-short questions, modeled conceptually on the platform's
  existing `treatment` A/B arm (which already correlates with longer
  questions — see Recommendation 2).
- **Metric to monitor:** Average `word_count` and quality-answer rate for
  the `very_short` bucket, pre/post rollout; question-submission volume
  (to catch unintended friction from the prompt).

---

**Recommendation 2 — Run a dedicated causal read on the A/B `variant`
before expanding it.**
- **Finding:** The `treatment` group shows longer questions, a higher
  quality rate, and higher engagement than `control` — but the trained
  model assigns `variant` itself almost no standalone importance.
- **Evidence:** Treatment vs. control: 48.1 vs 33.4 avg words, 71.6% vs
  67.7% quality rate, 6.71 vs 5.91 avg upvotes (§8); `variant`'s
  permutation importance is 0.0005, near noise level, once `word_count` is
  in the model (§13).
- **Product implication:** The treatment experience's benefit looks like
  it may run largely *through* length rather than being an independent
  effect — a mediation hypothesis, not (yet) a confirmed causal claim.
- **Recommended action:** Before expanding `treatment` platform-wide,
  run a formal causal/uplift analysis (confidence intervals, a mediation
  check on `word_count`) rather than relying on the descriptive group
  difference alone.
- **Metric to monitor:** Quality-answer rate and `engagement_score` by
  `variant`, evaluated with a pre-registered statistical test; word-count
  distribution by `variant` as the candidate mediating variable.

---

**Recommendation 3 — Gate any automated action to the model's reliable
zone; route everything else to humans.**
- **Finding:** The model is accurate at the length extremes but close to a
  coin flip on medium/long/very-long content — the bulk of content by volume.
- **Evidence:** Test accuracy by bucket: very_short 92.7%, short 82.9%,
  medium 45.3%, long 44.0%, very_long 54.9% (§14); overall test precision
  is only 0.349 (§12).
- **Product implication:** Trusting the model uniformly across all content
  lengths will produce roughly as many wrong calls as right ones in the
  medium/long range — an unacceptable basis for automated action there.
- **Recommended action:** Restrict any automated action (e.g.
  deprioritizing review-queue time) to the `very_short`/`short` buckets
  where accuracy is high; route `medium`/`long`/`very_long` predictions to
  a human-reviewed queue instead of automated action.
- **Metric to monitor:** Production precision/recall **by
  `word_count_bucket`**, not just in aggregate; reviewer override rate on
  model flags (a rising override rate signals model drift).

---

**Recommendation 4 — Don't over-invest in topic/country/device-specific
quality initiatives on this evidence.**
- **Finding:** Topic, country, and device each show only modest
  (a few percentage points) effects, both descriptively and in the model.
- **Evidence:** Topic quality-rate spread 68.4–71.4% (§8); country spread
  68.3–70.7%; device spread 69.4–69.7%; all near-zero standalone
  permutation importance except length and asker history (§13).
- **Product implication:** Segment-specific product campaigns (e.g. a
  topic-targeted quality push) are unlikely to be high-leverage compared
  to the platform-wide length intervention in Recommendation 1.
- **Recommended action:** Deprioritize segment-specific quality
  initiatives for now; revisit only if a larger or longer-window dataset
  shows a materially larger segment effect than observed here.
- **Metric to monitor:** Quality/engagement rate by topic/country/device,
  tracked on a quarterly monitoring cadence — to catch emerging outliers,
  not to justify near-term investment.

---

**Recommendation 5 — Prioritize capturing raw question (and answer) text.**
- **Finding:** The absence of raw text is very likely the single biggest
  ceiling on model performance.
- **Evidence:** The model has essentially one usable signal (§13); test
  PR-AUC (0.395) leaves most variance unexplained; the highest-confidence
  false positive/negative examples (§14) show two questions of similar
  length can have opposite outcomes — information the model structurally
  cannot see without text.
- **Product implication:** Data-engineering investment in capturing and
  retaining raw text (with appropriate privacy/retention review) would
  likely unlock the next meaningful jump in prediction quality.
- **Recommended action:** Prioritize this as a data-infrastructure
  workstream; once available, extend this pipeline with the TF-IDF/
  embedding/readability features documented as "not derivable" in §7.
- **Metric to monitor:** PR-AUC/ROC-AUC of a text-augmented model vs. this
  metadata-only baseline, measured on the same chronological test
  methodology used here — the cleanest way to quantify that data
  investment's return.

---

**Recommendation 6 — Pilot a lightweight "asker history" signal for
reviewers, at small scale.**
- **Finding:** Askers with a stronger track record of quality answers show
  a modest but real positive relationship with future engagement.
- **Evidence:** `asker_prior_quality_rate` is the second-most-important
  feature by permutation importance (0.0101) — the only other feature with
  a non-negligible standalone effect, though far smaller than length's
  0.1295 (§13).
- **Product implication:** There may be a small "trusted asker" signal
  worth surfacing lightly in product/review workflows, but the effect is
  too modest to justify a major reputation-system build on this evidence
  alone.
- **Recommended action:** Test surfacing asker history as **additional
  context for human reviewers** (not an automated score), at small scale,
  before considering any larger investment.
- **Metric to monitor:** Reviewer decision accuracy and time-to-decision,
  with vs. without asker-history context, in a small controlled test.

## 18. Limitations

- **Dataset limitations.** A single, synthetic ~9-week extract (~37.7k
  questions, ~12k users) from one Q&A platform. Activity is logged at
  **day-level, not timestamp-level**, granularity, which alone rules out
  any genuine "early engagement" feature anywhere in this project.
- **Potential selection bias.** No documentation of how this ~9-week
  window or these ~12,000 users were sampled from a larger population; the
  data may not represent longer-term or seasonal platform behavior, and
  users who asked zero questions in the window contribute no per-question
  rows to the modeling table at all.
- **Target-definition limitations.** `is_high_engagement` depends on
  `engagement_score`'s **equal weighting** of min-max-scaled views,
  answers, and upvotes — a documented judgment call (§9), not a
  platform-given ground truth. A different weighting (e.g. weighting
  upvotes higher as a more deliberate signal than passive views) could
  shift which questions are labeled positive and, in turn, model
  conclusions. `got_quality_answer`, while platform-recorded, has no
  documented underlying definition of "quality" available in this data.
- **Class imbalance.** `is_high_engagement` ~75/25 (3:1), `got_quality_answer`
  ~70/30 (2.3:1) — moderate, not severe. Addressed via
  `class_weight="balanced"`/`scale_pos_weight` and PR-AUC-first evaluation
  rather than resampling, but this still means default-threshold precision
  is limited (0.349 on test) — a real deployment should choose its
  decision threshold based on the actual cost of false positives vs. false
  negatives in that specific use case, not default to 0.5.
- **Model limitations.** Test PR-AUC 0.395 vs. a 0.255 floor is a real but
  moderate lift, leaving most variance unexplained. All four trained
  models are within ~0.02 PR-AUC of each other — the ceiling here is
  feature-driven, not algorithm-driven. Hyperparameters were lightly,
  manually tuned on validation PR-AUC, not exhaustively searched.
  Probability **calibration was not assessed** — predicted probabilities
  are not verified to be well-calibrated, which matters if a real
  deployment wants to use the raw score (not just above/below a
  threshold). No serving latency, drift-monitoring, or A/A stability
  testing was performed — this is a validated **offline baseline**, not a
  production-ready system.
- **Potential leakage risks.** The primary leakage risks (`num_answers`,
  `num_upvotes`, `num_views`, unique-actor counts, and the raw
  `experiment.csv` per-user aggregates) were identified and excluded with
  an explicit audit (§6). One **residual, lower-severity** risk: features
  like `word_count_z_within_topic` (topic mean/std) and `engagement_score`
  (min-max scaling) are computed once over the **full** dataset in this
  project's notebooks, rather than being re-fit on train-only data inside
  each split/fold. For a production pipeline, these dataset-level
  statistics should be refit on training data only at each retraining
  cycle to fully eliminate any residual information flow from
  validation/test-period data into training.
- **Generalization limitations.** A single ~9-week synthetic extract from
  one platform. The very strong dominance of one feature (length) and
  generally flat segment effects are consistent with either a genuinely
  simple underlying engagement process, or a dataset with less naturalistic
  complexity than a real production Q&A platform would have. Results here
  should be read as **directionally instructive**, not a guarantee of
  similar performance on live, larger, or differently-structured platform data.
- **Correlation vs. causation.** Every relationship in this project is
  predictive/associational, with one partial exception: `variant` is
  randomized and pre-assigned, making its descriptive group difference
  (§8) a legitimate *candidate* for a causal read — but this project only
  reports the observed difference, without confidence intervals,
  multiple-comparison adjustment, or a formal mediation analysis of how
  much of that effect runs through `word_count` specifically (§13
  flags this as a hypothesis, not a proof). Every other feature (topic,
  country, `asker_prior_quality_rate`, etc.) is purely observational — for
  example, "longer questions get better answers" is equally consistent
  with more engaged or skilled askers self-selecting into writing longer
  questions, not length itself causing the outcome. **No causal claims
  should be drawn from this analysis**, and no automated, high-impact
  product decision should be justified on causal grounds without a
  dedicated causal-inference study.

## 19. Tech Stack

- **Language:** Python 3.10+
- **Data manipulation:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Machine learning:** scikit-learn (Logistic Regression, Random Forest,
  HistGradientBoostingClassifier, preprocessing pipelines, metrics,
  permutation importance), XGBoost
- **Model interpretation:** SHAP
- **Environment:** Jupyter notebooks + a mirrored, importable `src/` package
- Full pinned list in [`requirements.txt`](requirements.txt)

## 20. Project Structure

```text
qa-content-quality-prediction/
│
├── data/
│   ├── modeling_dataset.csv        # Part 1 output (37,719 x 27) — cleaned, leakage-audited
│   ├── modeling_dataset_v2.csv     # Part 2 output (37,719 x 31) — + NLP & topic-history features
│   └── data_quality_summary.md     # Part 1 written data-quality summary
│   # Raw extracts (users.csv, questions.csv, user_activity.csv,
│   # experiment.csv) are not redistributed here — see §4.
│
├── notebooks/
│   ├── 01_data_profiling.ipynb          # Inspection, quality checks, leakage audit, cleaning
│   ├── 02_eda_feature_engineering.ipynb # Deeper EDA, NLP features, target definition
│   ├── 03_modeling.ipynb                # Split, preprocessing, 4 models, comparison, selection
│   └── 04_model_interpretation.ipynb    # Coefficients, permutation importance, SHAP, error analysis
│
├── src/
│   ├── data_cleaning.py        # Load raw data, quality checks, leakage-safe feature construction
│   ├── feature_engineering.py  # NLP-derived features, target definition, final feature set
│   ├── modeling.py             # Chronological split, preprocessing pipelines, model definitions
│   └── evaluation.py           # Metrics, model selection, interpretation, error analysis helpers
│
├── outputs/
│   ├── figures/                # 24 exported PNGs (6 Part-1 + 9 Part-2 + 9 Part-3 figures)
│   └── tables/                 # model_comparison.csv, permutation_importance.csv,
│                                # accuracy_by_length_bucket.csv, test_set_error_breakdown.csv,
│                                # topic_summary.csv, quality_by_word_count_bucket.csv,
│                                # variant_comparison.csv, quality_rate_by_country.csv
│
├── README.md
└── requirements.txt
```

**Reproducing this project:**
```bash
pip install -r requirements.txt

# Interactively, notebook by notebook:
jupyter notebook notebooks/

# Or from the command line, via src/:
cd src
python data_cleaning.py          # requires raw CSVs in ../data/ (see §4)
python feature_engineering.py    # runs against ../data/modeling_dataset.csv
python modeling.py               # runs against ../data/modeling_dataset_v2.csv
python evaluation.py             # trains + evaluates + selects the best model
```

Notebooks `02`–`04` and all of `src/` run standalone against the processed
CSVs already included in `data/`; only `01_data_profiling.ipynb` (and
`src/data_cleaning.py`) requires the raw extracts.
