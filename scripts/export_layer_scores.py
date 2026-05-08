"""Export train/test per-layer sentiment scores for combination diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


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


def build_layer_scores(
    gathered_data: dict[str, Any],
    directions_data: dict[str, Any],
    validation_data: dict[str, Any],
) -> dict[str, Any]:
    train_hidden_states = require_tensor(gathered_data, "hidden_states")
    train_labels = require_tensor(gathered_data, "labels")
    directions = require_tensor(directions_data, "directions")
    test_scores = require_tensor(validation_data, "scores")
    test_labels = require_tensor(validation_data, "labels")

    train_scores = score_hidden_states(train_hidden_states, directions)
    if train_scores.dim() != 2 or test_scores.dim() != 2:
        raise ValueError("train_scores and test_scores must have shape [num_examples, num_layers]")
    if train_scores.shape[1] != test_scores.shape[1]:
        raise ValueError("train and test scores must have the same number of layers")
    if train_scores.shape[0] != train_labels.shape[0]:
        raise ValueError("train scores and labels must have the same number of examples")
    if test_scores.shape[0] != test_labels.shape[0]:
        raise ValueError("test scores and labels must have the same number of examples")

    return {
        "train_scores": train_scores,
        "test_scores": test_scores,
        "train_labels": train_labels.to(torch.long),
        "test_labels": test_labels.to(torch.long),
        "layers": list(range(train_scores.shape[1])),
        "metadata": {
            "gathered": gathered_data.get("metadata", {}),
            "directions": directions_data.get("metadata", {}),
            "validation_details": validation_data.get("metadata", {}),
            "score_rule": "hidden[layer] dot direction[layer]",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gathered", required=True)
    parser.add_argument("--directions", required=True)
    parser.add_argument("--validation-details", required=True)
    parser.add_argument(
        "--output",
        default="outputs/sst2_qwen2_0_5b_layer_scores.pt",
    )
    return parser.parse_args()


def load_dict(path: str) -> dict[str, Any]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a dictionary")
    return data


def main() -> None:
    args = parse_args()
    output = build_layer_scores(
        gathered_data=load_dict(args.gathered),
        directions_data=load_dict(args.directions),
        validation_data=load_dict(args.validation_details),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)


if __name__ == "__main__":
    main()
