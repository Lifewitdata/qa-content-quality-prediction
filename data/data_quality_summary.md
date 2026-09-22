# Data Quality & Preparation Summary

## Data-quality summary
- 4 clean, referentially-consistent CSV extracts: 12,000 users, 37,719
  questions, 215,370 activity events, 12,000 experiment rows.
- No duplicate IDs or full-row duplicates in any file.
- Missing values are limited to `experiment.csv` (`avg_word_count`,
  `quality_answer_rate`), and are structurally explained: users with zero
  questions asked have an undefined average/rate.
- All category values are already consistent (no casing/whitespace variants
  found); `users.csv` and `experiment.csv` category values agree per-user.
- All dates are internally consistent (no question predates its asker's
  signup, no activity predates its question). Activity events are logged at
  day-level granularity and always fall on the question's creation day.
- `questions.csv`'s own engagement counters (`num_views`/`num_answers`/
  `num_upvotes`) do **not** match an independent aggregation of
  `user_activity.csv` - the two are separate signal sources, not duplicates
  of one another.
- A handful (~445) of high-upvote outliers exist and look like
  genuine viral questions rather than data errors (no negative/impossible values).

## Cleaned dataset structure
- Final table: `modeling_dataset.csv`, one row per `question_id`
  (37,719 rows x 27 columns).
- Built by joining `questions` (base) with asker attributes from `users`,
  `variant` from `experiment`, and per-question unique-actor counts
  aggregated from `user_activity`.

## Important variables
- Content quality: `got_quality_answer`, `word_count`
- Engagement: `num_views`, `num_answers`, `num_upvotes`, `engagement_score`
- Context: `topic`, `asker_country`, `asker_device_type`, `variant`,
  `asker_account_age_at_question_days`, `asker_prior_quality_rate`

## Potential target variable
- Primary: `got_quality_answer` (binary classification)
- Secondary: `engagement_score` / `num_upvotes` (regression)

## Potential predictive features (leakage-safe)
`topic`, `word_count`, `word_count_bucket`, `created_dow`,
`created_is_weekend`, `asker_country`, `asker_device_type`, `variant`,
`asker_account_age_at_question_days`, `account_age_bucket`,
`asker_prior_questions_count`, `asker_prior_quality_rate`.

## Data leakage risks
- `num_answers` is a near-perfect proxy for `got_quality_answer` (0 counter-
  examples found) - exclude when predicting quality.
- `num_views`, `num_upvotes`, and all `user_activity`-derived per-question
  counts are concurrent/post-outcome signals (no timestamp-level data exists
  to isolate a legitimate "before the outcome" window) - exclude as features
  for the quality target; usable only as alternate engagement targets.
- `experiment.quality_answer_rate` / `avg_word_count` are aggregated across
  a user's questions and can leak the row's own outcome - replaced with a
  strictly-prior-in-time `asker_prior_quality_rate` instead.

## Recommended modeling approach
1. **Quality classification**: predict `got_quality_answer` from the
   leakage-safe feature set above using a tree-based classifier (Random
   Forest / Gradient Boosting) with class-imbalance-aware evaluation
   (ROC-AUC, PR-AUC) and topic/country as categorical (one-hot or target
   encoding with proper cross-validation to avoid leakage).
2. **Engagement regression**: predict `engagement_score` or
   `log1p(num_upvotes)` from the same leakage-safe feature set, as a
   separate model, since engagement counts themselves cannot be used as
   quality-classifier inputs.
3. Evaluate the A/B `variant` effect on both targets explicitly (it is a
   randomized, pre-assigned feature, well suited to a causal/uplift read as
   well as a plain predictive feature).
4. Once raw question text is available, extend the feature set with
   TF-IDF/embeddings per the NLP-readiness notes above.
