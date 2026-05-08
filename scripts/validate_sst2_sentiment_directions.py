"""Validate SST-2 sentiment directions on withheld examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent))
from gather_sst2_hidden_states import CAPTURE_METHODS, collect_hidden_states  # noqa: E402

DATASET_NAME = "stanfordnlp/sst2"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def require_tensor(data: dict[str, Any], key: str) -> torch.Tensor:
    value = data.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected {key!r} to be a tensor")
    return value


def score_hidden_states(hidden_states: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [num_examples, num_layers, hidden_size]")
    if directions.dim() != 2:
        raise ValueError("directions must have shape [num_layers, hidden_size]")
    if hidden_states.shape[1:] != directions.shape:
        raise ValueError("hidden_states layer/hidden dimensions must match directions")

    return (hidden_states * directions.unsqueeze(0)).sum(dim=2)


def predictions_from_scores(scores: torch.Tensor) -> torch.Tensor:
    return (scores > 0).to(torch.long)


def summarize_scores(scores: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    if scores.dim() != 2:
        raise ValueError("scores must have shape [num_examples, num_layers]")
    if labels.dim() != 1:
        raise ValueError("labels must have shape [num_examples]")
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("scores and labels must have the same number of examples")

    predictions = predictions_from_scores(scores)
    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_count = int(positive_mask.sum().item())
    negative_count = int(negative_mask.sum().item())

    layers = []
    for layer in range(scores.shape[1]):
        layer_predictions = predictions[:, layer]
        correct = layer_predictions == labels
        positive_correct = layer_predictions[positive_mask] == labels[positive_mask]
        negative_correct = layer_predictions[negative_mask] == labels[negative_mask]
        layers.append(
            {
                "layer": layer,
                "accuracy": float(correct.float().mean().item()),
                "positive_accuracy": (
                    float(positive_correct.float().mean().item()) if positive_count else None
                ),
                "negative_accuracy": (
                    float(negative_correct.float().mean().item()) if negative_count else None
                ),
                "mean_positive_score": (
                    float(scores[positive_mask, layer].mean().item()) if positive_count else None
                ),
                "mean_negative_score": (
                    float(scores[negative_mask, layer].mean().item()) if negative_count else None
                ),
            }
        )

    best_layer = max(layers, key=lambda layer_summary: layer_summary["accuracy"])
    return {
        "num_examples": int(labels.shape[0]),
        "counts": {
            "positive": positive_count,
            "negative": negative_count,
        },
        "layers": layers,
        "best_layer": {
            "layer": best_layer["layer"],
            "accuracy": best_layer["accuracy"],
        },
    }


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", required=True)
    parser.add_argument("--gathered", required=True)
    parser.add_argument(
        "--details-output",
        default="outputs/sst2_qwen2_0_5b_validation_details.pt",
    )
    parser.add_argument(
        "--summary-output",
        default="outputs/sst2_qwen2_0_5b_validation_summary.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--capture-method", choices=CAPTURE_METHODS, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    directions_data = torch.load(args.directions, map_location="cpu")
    if not isinstance(directions_data, dict):
        raise ValueError("directions file must contain a dictionary")
    directions = require_tensor(directions_data, "directions")
    if directions.dim() != 2:
        raise ValueError("directions must have shape [num_layers, hidden_size]")

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
    capture_method = args.capture_method or str(
        gathered_metadata.get("capture_method", "output-hidden-states"),
    )

    dataset = load_dataset(DATASET_NAME, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    directions = directions.to(device)

    score_batches: list[torch.Tensor] = []
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

        with torch.no_grad():
            hidden_states, layer_indexing = collect_hidden_states(
                model=model,
                tokens=tokens,
                capture_method=capture_method,
            )

        token_indices = last_non_padding_indices(tokens["attention_mask"])
        batch_positions = torch.arange(len(batch_texts), device=device)

        per_layer = [layer[batch_positions, token_indices, :].detach() for layer in hidden_states]
        batch_hidden_states = torch.stack(per_layer, dim=1)
        score_batches.append(score_hidden_states(batch_hidden_states, directions).cpu())
        labels.extend(batch_labels)
        texts.extend(batch_texts)
        source_indices.extend(batch_indices)

    scores = torch.cat(score_batches, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    predictions = predictions_from_scores(scores)
    summary = summarize_scores(scores, label_tensor)

    details: dict[str, Any] = {
        "scores": scores,
        "labels": label_tensor,
        "predictions": predictions,
        "texts": texts,
        "source_indices": source_indices,
        "metadata": {
            "directions_metadata": directions_data.get("metadata", {}),
            "gathered_metadata": gathered_metadata,
            "dataset": DATASET_NAME,
            "source_split": "train",
            "model": MODEL_NAME,
            "capture_method": capture_method,
            "layer_indexing": layer_indexing,
            "prediction_rule": "score > 0 predicts positive",
        },
    }

    details_path = Path(args.details_output)
    details_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(details, details_path)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
