"""Gather SST-2 hidden states from Qwen2.5-0.5B-Instruct."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_NAME = "stanfordnlp/sst2"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LABEL_NAMES = {"0": "negative", "1": "positive"}
CAPTURE_METHODS = ("output-hidden-states", "forward-hooks")


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


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("Expected model.model.layers to be a torch.nn.ModuleList")
    return layers


def layer_hidden_from_output(output: Any) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    if not isinstance(hidden, torch.Tensor):
        raise ValueError("Expected decoder layer output to contain a tensor hidden state")
    return hidden


def capture_decoder_block_outputs(
    model: nn.Module,
    tokens: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    captures: list[torch.Tensor | None] = [None for _ in decoder_layers(model)]
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(index: int) -> Any:
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            captures[index] = layer_hidden_from_output(output)

        return hook

    for index, layer in enumerate(decoder_layers(model)):
        handles.append(layer.register_forward_hook(make_hook(index)))

    try:
        model(**tokens)
    finally:
        for handle in handles:
            handle.remove()

    missing = [index for index, capture in enumerate(captures) if capture is None]
    if missing:
        raise RuntimeError(f"Did not capture decoder block outputs for layers: {missing}")
    return [capture for capture in captures if capture is not None]


def collect_hidden_states(
    model: nn.Module,
    tokens: dict[str, torch.Tensor],
    capture_method: str,
) -> tuple[list[torch.Tensor], str]:
    if capture_method == "output-hidden-states":
        outputs = model(**tokens, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        return list(hidden_states), (
            "output_hidden_states index 0 is embedding output; index N is after decoder block N-1"
        )

    if capture_method == "forward-hooks":
        return capture_decoder_block_outputs(model, tokens), (
            "forward hook index N is raw decoder block N output; for non-final blocks, compare "
            "to output_hidden_states[N+1]. The final hook output is before final model norm."
        )

    raise ValueError(f"Unknown capture method: {capture_method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/sst2_qwen2_0_5b_hidden_states.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--capture-method", choices=CAPTURE_METHODS, default="output-hidden-states")
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
            hidden_states, layer_indexing = collect_hidden_states(
                model=model,
                tokens=tokens,
                capture_method=args.capture_method,
            )

        token_indices = last_non_padding_indices(tokens["attention_mask"])
        batch_positions = torch.arange(len(batch_texts), device=device)
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
            "capture_method": args.capture_method,
            "layer_indexing": layer_indexing,
            "label_names": LABEL_NAMES,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, output_path)


if __name__ == "__main__":
    main()
