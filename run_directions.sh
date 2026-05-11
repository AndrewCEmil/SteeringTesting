echo "Mean-Difference"
uv run python scripts/compute_sst2_sentiment_directions.py \
      --input outputs/smoke_hidden_states_hooks.pt \
      --output outputs/smoke_sentiment_mean_diff_hooks.pt \
      --probe-type mean_diff

uv run python scripts/validate_sst2_sentiment_directions.py \
      --directions outputs/smoke_sentiment_mean_diff_hooks.pt \
      --gathered outputs/smoke_hidden_states_hooks.pt \
      --details-output outputs/smoke_validation_details_mean_diff_hooks.pt \
      --summary-output outputs/smoke_validation_summary_mean_diff_hooks.json \
      --batch-size 1 \
      --max-examples 1000 \
      --device mps \
      --capture-method forward-hooks

# Logistic Regression
echo "Logistic Regression"
uv run python scripts/compute_sst2_sentiment_directions.py \
      --input outputs/smoke_hidden_states_hooks.pt \
      --output outputs/smoke_sentiment_logreg_hooks.pt \
      --probe-type logistic_regression \
      --c 1.0 \
      --max-iter 1000

uv run python scripts/validate_sst2_sentiment_directions.py \
      --directions outputs/smoke_sentiment_logreg_hooks.pt \
      --gathered outputs/smoke_hidden_states_hooks.pt \
      --details-output outputs/smoke_validation_details_logreg_hooks.pt \
      --summary-output outputs/smoke_validation_summary_logreg_hooks.json \
      --batch-size 1 \
      --max-examples 1000 \
      --device mps \
      --capture-method forward-hooks

# Linear SVM
echo "Linear SVM"
uv run python scripts/compute_sst2_sentiment_directions.py \
      --input outputs/smoke_hidden_states_hooks.pt \
      --output outputs/smoke_sentiment_svm_hooks.pt \
      --probe-type linear_svm \
      --c 1.0 \
      --max-iter 5000

uv run python scripts/validate_sst2_sentiment_directions.py \
      --directions outputs/smoke_sentiment_svm_hooks.pt \
      --gathered outputs/smoke_hidden_states_hooks.pt \
      --details-output outputs/smoke_validation_details_svm_hooks.pt \
      --summary-output outputs/smoke_validation_summary_svm_hooks.json \
      --batch-size 1 \
      --max-examples 1000 \
      --device mps \
      --capture-method forward-hooks

# Whitened mean-diff
echo "Whitened Mean-Difference"
uv run python scripts/compute_sst2_sentiment_directions.py \
      --input outputs/smoke_hidden_states_hooks.pt \
      --output outputs/smoke_sentiment_whitened_hooks.pt \
      --probe-type whitened_mean_diff \
      --whitening-eps 1e-4

uv run python scripts/validate_sst2_sentiment_directions.py \
      --directions outputs/smoke_sentiment_whitened_hooks.pt \
      --gathered outputs/smoke_hidden_states_hooks.pt \
      --details-output outputs/smoke_validation_details_whitened_hooks.pt \
      --summary-output outputs/smoke_validation_summary_whitened_hooks.json \
      --batch-size 1 \
      --max-examples 1000 \
      --device mps \
      --capture-method forward-hooks

# Low-rank subspace, rank 3
echo "Low-Rank Subspace (Rank 3)"
uv run python scripts/compute_sst2_sentiment_directions.py \
      --input outputs/smoke_hidden_states_hooks.pt \
      --output outputs/smoke_sentiment_low_rank_r3_hooks.pt \
      --probe-type low_rank_subspace \
      --rank 3

uv run python scripts/validate_sst2_sentiment_directions.py \
      --directions outputs/smoke_sentiment_low_rank_r3_hooks.pt \
      --gathered outputs/smoke_hidden_states_hooks.pt \
      --details-output outputs/smoke_validation_details_low_rank_r3_hooks.pt \
      --summary-output outputs/smoke_validation_summary_low_rank_r3_hooks.json \
      --batch-size 1 \
      --max-examples 1000 \
      --device mps \
      --capture-method forward-hooks
