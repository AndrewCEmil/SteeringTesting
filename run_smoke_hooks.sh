echo "Computing hidden states with forward hooks"
uv run python scripts/gather_sst2_hidden_states.py \
    --output outputs/smoke_hidden_states_hooks.pt \
    --batch-size 1 \
    --max-examples 1000 \
    --device cpu \
    --capture-method forward-hooks


echo "Gathering directions"
uv run python scripts/compute_sst2_sentiment_directions.py \
    --input outputs/smoke_hidden_states_hooks.pt \
    --output outputs/smoke_sentiment_directions_hooks.pt


echo "Validating directions"
uv run python scripts/validate_sst2_sentiment_directions.py \
    --directions outputs/smoke_sentiment_directions_hooks.pt \
    --gathered outputs/smoke_hidden_states_hooks.pt \
    --details-output outputs/smoke_validation_details_hooks.pt \
    --summary-output outputs/smoke_validation_summary_hooks.json \
    --batch-size 1 \
    --max-examples 1000 \
    --device cpu \
    --capture-method forward-hooks
