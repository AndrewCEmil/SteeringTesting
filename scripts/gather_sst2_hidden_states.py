"""Gather SST-2 hidden states from Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "stanfordnlp/sst2"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LABEL_NAMES = {"0": "negative", "1": "positive"}


def split_indices(size: int, heldout_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("--heldout-fraction must be greater than 0 and less than 1")

    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    heldout_size = int(size * heldout_fraction)
    heldout_indices = sorted(indices[:heldout_size])
    gather_indices = sorted(indices[heldout_size:])
    return gather_indices, heldout_indices


def last_non_padding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1) - 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/sst2_qwen2_0_5b_hidden_states.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(DATASET_NAME, split="train")
    gather_indices, heldout_indices = split_indices(
        size=len(dataset),
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
    )
    if args.max_examples is not None:
        gather_indices = gather_indices[: args.max_examples]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    hidden_state_batches: list[torch.Tensor] = []
    texts: list[str] = []
    labels: list[int] = []
    source_indices: list[int] = []

    for start in range(0, len(gather_indices), args.batch_size):
        batch_indices = gather_indices[start : start + args.batch_size]
        batch_items = dataset.select(batch_indices)
        batch_texts = list(batch_items["sentence"])
        batch_labels = [int(label) for label in batch_items["label"]]

        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        token_indices = last_non_padding_indices(tokens["attention_mask"])
        batch_positions = torch.arange(len(batch_texts), device=device)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")

        per_layer = [
            layer[batch_positions, token_indices, :].detach().cpu() for layer in hidden_states
        ]
        hidden_state_batches.append(torch.stack(per_layer, dim=1))
        texts.extend(batch_texts)
        labels.extend(batch_labels)
        source_indices.extend(batch_indices)

    output: dict[str, Any] = {
        "hidden_states": torch.cat(hidden_state_batches, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
        "source_indices": source_indices,
        "heldout_indices": heldout_indices,
        "metadata": {
            "dataset": DATASET_NAME,
            "source_split": "train",
            "model": MODEL_NAME,
            "seed": args.seed,
            "heldout_fraction": args.heldout_fraction,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "label_names": LABEL_NAMES,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)


if __name__ == "__main__":
    main()
