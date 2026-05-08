uv run python scripts/export_layer_scores.py \
    --gathered outputs/smoke_hidden_states.pt \
    --directions outputs/smoke_sentiment_directions.pt \
    --validation-details outputs/_validation_details.pt \
    --output outputs/smoke_layer_scores.pt

uv run python scripts/combine_layer_scores.py \
--input outputs/smoke_layer_scores.pt \
--layers 12 13 14 15 16 \
--output outputs/smoke_layer_combination_middle_clean.json

uv run python scripts/combine_layer_scores.py \
--input outputs/smoke_layer_scores.pt \
--layers 12 13 14 15 16 20 22 23 \
--output outputs/smoke_layer_combination_middle_plus_late.json

