"""Evaluate a simple sentiment-direction residual stream intervention."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from mechinterp.sentiment_probes import component_directions, primary_directions

sys.path.append(str(Path(__file__).parent))
from gather_sst2_hidden_states import (  # noqa: E402
    DATASET_NAME,
    MODEL_NAME,
    decoder_layers,
    last_non_padding_indices,
    layer_hidden_from_output,
)


def predictions_from_scores(scores: torch.Tensor) -> torch.Tensor:
    return (scores > 0).to(torch.long)


def apply_direction_to_token_positions(
    output: Any,
    token_indices: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> tuple[Any, torch.Tensor, torch.Tensor]:
    hidden = layer_hidden_from_output(output)
    modified_hidden = hidden.clone()
    batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
    original_values = hidden[batch_positions, token_indices, :].detach()
    modified_hidden[batch_positions, token_indices, :] += alpha * direction
    modified_values = modified_hidden[batch_positions, token_indices, :].detach()
    if isinstance(output, tuple):
        return (modified_hidden, *output[1:]), original_values, modified_values
    return modified_hidden, original_values, modified_values


def run_intervened_forward(
    model: nn.Module,
    tokens: dict[str, torch.Tensor],
    hook_block_index: int,
    direction: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_indices = last_non_padding_indices(tokens["attention_mask"])
    layers = decoder_layers(model)
    if hook_block_index < 0 or hook_block_index >= len(layers):
        raise ValueError(f"hook_block_index must be between 0 and {len(layers) - 1}")

    captured_original: torch.Tensor | None = None
    captured_modified: torch.Tensor | None = None

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        nonlocal captured_original, captured_modified
        modified_output, original_values, modified_values = apply_direction_to_token_positions(
            output=output,
            token_indices=token_indices,
            direction=direction,
            alpha=alpha,
        )
        captured_original = original_values
        captured_modified = modified_values
        return modified_output

    handle = layers[hook_block_index].register_forward_hook(hook)
    try:
        model(**tokens)
    finally:
        handle.remove()

    if captured_original is None or captured_modified is None:
        raise RuntimeError(f"Did not capture intervention values for hook block {hook_block_index}")
    return captured_original, captured_modified


def summarize_alpha(
    alpha: float,
    baseline_scores: torch.Tensor,
    intervened_scores: torch.Tensor,
    baseline_predictions: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    score_delta = intervened_scores - baseline_scores
    delta_error = (score_delta - alpha).abs()
    predictions = predictions_from_scores(intervened_scores)
    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_count = int(positive_mask.sum().item())
    negative_count = int(negative_mask.sum().item())

    return {
        "alpha": float(alpha),
        "accuracy": float((predictions == labels).float().mean().item()),
        "positive_accuracy": (
            float((predictions[positive_mask] == labels[positive_mask]).float().mean().item())
            if positive_count
            else None
        ),
        "negative_accuracy": (
            float((predictions[negative_mask] == labels[negative_mask]).float().mean().item())
            if negative_count
            else None
        ),
        "mean_score_delta": float(score_delta.mean().item()),
        "expected_score_delta": float(alpha),
        "mean_abs_delta_error": float(delta_error.mean().item()),
        "max_abs_delta_error": float(delta_error.max().item()),
        "flip_rate": float((predictions != baseline_predictions).float().mean().item()),
        "positive_prediction_rate": float(predictions.float().mean().item()),
        "mean_positive_score": (
            float(intervened_scores[positive_mask].mean().item()) if positive_count else None
        ),
        "mean_negative_score": (
            float(intervened_scores[negative_mask].mean().item()) if negative_count else None
        ),
    }


def build_example_details(
    alpha: float,
    baseline_scores: torch.Tensor,
    intervened_scores: torch.Tensor,
    labels: torch.Tensor,
    texts: Sequence[str],
    source_indices: Sequence[int],
) -> list[dict[str, Any]]:
    baseline_predictions = predictions_from_scores(baseline_scores)
    intervened_predictions = predictions_from_scores(intervened_scores)
    rows = zip(
        source_indices,
        labels.tolist(),
        texts,
        baseline_scores.tolist(),
        intervened_scores.tolist(),
        baseline_predictions.tolist(),
        intervened_predictions.tolist(),
        strict=True,
    )
    details = []
    for row in rows:
        (
            source_index,
            label,
            text,
            baseline_score,
            intervened_score,
            baseline_prediction,
            intervened_prediction,
        ) = row
        details.append(
            {
                "source_index": int(source_index),
                "label": int(label),
                "text": text,
                "alpha": float(alpha),
                "baseline_score": float(baseline_score),
                "intervened_score": float(intervened_score),
                "score_delta": float(intervened_score - baseline_score),
                "baseline_prediction": int(baseline_prediction),
                "intervened_prediction": int(intervened_prediction),
                "flipped": bool(baseline_prediction != intervened_prediction),
            },
        )
    return details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", default="outputs/smoke_sentiment_directions.pt")
    parser.add_argument("--gathered", default="outputs/smoke_hidden_states.pt")
    parser.add_argument("--output", default="outputs/smoke_sentiment_intervention_layer14.json")
    parser.add_argument("--hook-block-index", type=int, default=13)
    parser.add_argument("--direction-layer-index", type=int, default=14)
    parser.add_argument("--component-index", type=int, default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--include-details", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    directions_data = torch.load(args.directions, map_location="cpu")
    if not isinstance(directions_data, dict):
        raise ValueError("directions file must contain a dictionary")
    if args.component_index is None:
        directions = primary_directions(directions_data)
    else:
        components = component_directions(directions_data)
        if args.component_index < 0 or args.component_index >= components.shape[1]:
            raise ValueError(f"component_index must be between 0 and {components.shape[1] - 1}")
        directions = components[:, args.component_index, :]
    if args.direction_layer_index < 0 or args.direction_layer_index >= directions.shape[0]:
        raise ValueError(
            f"direction_layer_index must be between 0 and {directions.shape[0] - 1}",
        )
    direction = directions[args.direction_layer_index].to(device)

    gathered_data = torch.load(args.gathered, map_location="cpu")
    if not isinstance(gathered_data, dict):
        raise ValueError("gathered file must contain a dictionary")
    heldout_indices = gathered_data.get("heldout_indices")
    if not isinstance(heldout_indices, list):
        raise ValueError("gathered file must contain heldout_indices")
    if args.max_examples is not None:
        heldout_indices = heldout_indices[: args.max_examples]
    if not heldout_indices:
        raise ValueError("withheld set is empty")

    gathered_metadata = gathered_data.get("metadata", {})
    if not isinstance(gathered_metadata, dict):
        gathered_metadata = {}
    max_length = args.max_length or int(gathered_metadata.get("max_length", 128))

    dataset = load_dataset(DATASET_NAME, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    alpha_scores: dict[float, list[torch.Tensor]] = {alpha: [] for alpha in args.alphas}
    baseline_scores_by_alpha: dict[float, list[torch.Tensor]] = {alpha: [] for alpha in args.alphas}
    labels: list[int] = []
    texts: list[str] = []
    source_indices: list[int] = []

    for start in range(0, len(heldout_indices), args.batch_size):
        batch_indices = heldout_indices[start : start + args.batch_size]
        batch_items = dataset.select(batch_indices)
        batch_texts = list(batch_items["sentence"])
        batch_labels = [int(label) for label in batch_items["label"]]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}

        for alpha in args.alphas:
            with torch.no_grad():
                original_values, modified_values = run_intervened_forward(
                    model=model,
                    tokens=tokens,
                    hook_block_index=args.hook_block_index,
                    direction=direction,
                    alpha=alpha,
                )
            baseline_scores = (original_values * direction.unsqueeze(0)).sum(dim=1)
            intervened_scores = (modified_values * direction.unsqueeze(0)).sum(dim=1)
            baseline_scores_by_alpha[alpha].append(baseline_scores.cpu())
            alpha_scores[alpha].append(intervened_scores.cpu())

        labels.extend(batch_labels)
        texts.extend(batch_texts)
        source_indices.extend(batch_indices)

    label_tensor = torch.tensor(labels, dtype=torch.long)
    baseline_alpha = 0.0 if 0.0 in args.alphas else args.alphas[0]
    baseline_scores = torch.cat(alpha_scores[baseline_alpha], dim=0)
    baseline_predictions = predictions_from_scores(baseline_scores)

    alpha_reports = []
    details_by_alpha: dict[str, list[dict[str, Any]]] = {}
    for alpha in args.alphas:
        intervened_scores = torch.cat(alpha_scores[alpha], dim=0)
        original_scores = torch.cat(baseline_scores_by_alpha[alpha], dim=0)
        alpha_reports.append(
            summarize_alpha(
                alpha=alpha,
                baseline_scores=original_scores,
                intervened_scores=intervened_scores,
                baseline_predictions=baseline_predictions,
                labels=label_tensor,
            ),
        )
        if args.include_details:
            details_by_alpha[str(alpha)] = build_example_details(
                alpha=alpha,
                baseline_scores=baseline_scores,
                intervened_scores=intervened_scores,
                labels=label_tensor,
                texts=texts,
                source_indices=source_indices,
            )

    report: dict[str, Any] = {
        "metadata": {
            "dataset": DATASET_NAME,
            "source_split": "train",
            "model": MODEL_NAME,
            "directions": args.directions,
            "gathered": args.gathered,
            "num_examples": len(labels),
            "batch_size": args.batch_size,
            "max_length": max_length,
            "hook_block_index": args.hook_block_index,
            "direction_layer_index": args.direction_layer_index,
            "component_index": args.component_index,
            "alphas": [float(alpha) for alpha in args.alphas],
            "baseline_alpha": float(baseline_alpha),
            "layer_indexing": (
                "hook block 13 corresponds to output_hidden_states[14] for the current Qwen "
                "decoder stack; intervention is applied to the last non-padding token only"
            ),
            "prediction_rule": "score > 0 predicts positive",
        },
        "alphas": alpha_reports,
    }
    if args.include_details:
        report["details_by_alpha"] = details_by_alpha

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
