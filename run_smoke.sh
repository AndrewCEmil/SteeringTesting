echo "Computing hidden states"
uv run python scripts/gather_sst2_hidden_states.py \
    --output outputs/smoke_hidden_states.pt \
    --batch-size 1 \
    --max-examples 1000 \
    --device cpu


echo "Gathering directions"
uv run python scripts/compute_sst2_sentiment_directions.py \
    --input outputs/smoke_hidden_states.pt \
    --output outputs/smoke_sentiment_directions.pt


echo "Validating directions"
uv run python scripts/validate_sst2_sentiment_directions.py \
    --directions outputs/smoke_sentiment_directions.pt \
    --gathered outputs/smoke_hidden_states.pt \
    --details-output outputs/smoke_validation_details.pt \
    --summary-output outputs/smoke_validation_summary.json \
    --batch-size 1 \
    --max-examples 1000 \
    --device cpu
