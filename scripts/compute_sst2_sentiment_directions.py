"""Compute simple SST-2 sentiment directions from gathered hidden states."""

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


def compute_directions(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int], list[int]]:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [num_examples, num_layers, hidden_size]")
    if labels.dim() != 1:
        raise ValueError("labels must have shape [num_examples]")
    if hidden_states.shape[0] != labels.shape[0]:
        raise ValueError("hidden_states and labels must have the same number of examples")

    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_count = int(positive_mask.sum().item())
    negative_count = int(negative_mask.sum().item())
    if positive_count == 0:
        raise ValueError("labels must include at least one positive example")
    if negative_count == 0:
        raise ValueError("labels must include at least one negative example")

    mean_positive = hidden_states[positive_mask].mean(dim=0)
    mean_negative = hidden_states[negative_mask].mean(dim=0)
    directions = mean_positive - mean_negative
    norms = directions.norm(dim=1, keepdim=True)
    nonzero_norms = norms.squeeze(dim=1) != 0
    normalized_directions = torch.zeros_like(directions)
    normalized_directions[nonzero_norms] = directions[nonzero_norms] / norms[nonzero_norms]

    counts = {"positive": positive_count, "negative": negative_count}
    zero_norm_layers = torch.nonzero(~nonzero_norms).squeeze(dim=1).tolist()
    return normalized_directions, mean_positive, mean_negative, counts, zero_norm_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="outputs/sst2_qwen2_0_5b_sentiment_directions.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.input, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("input file must contain a dictionary")

    hidden_states = require_tensor(data, "hidden_states")
    labels = require_tensor(data, "labels")
    directions, mean_positive, mean_negative, counts, zero_norm_layers = compute_directions(
        hidden_states,
        labels,
    )

    output: dict[str, Any] = {
        "directions": directions,
        "mean_positive": mean_positive,
        "mean_negative": mean_negative,
        "counts": counts,
        "metadata": {
            "source": data.get("metadata", {}),
            "analysis": "mean_positive_minus_mean_negative",
            "normalized": True,
            "zero_norm_layers": zero_norm_layers,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)


if __name__ == "__main__":
    main()
