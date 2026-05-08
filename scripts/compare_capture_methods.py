"""Compare output_hidden_states with decoder-block forward hook captures."""

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
from gather_sst2_hidden_states import (
    DATASET_NAME,
    MODEL_NAME,
    capture_decoder_block_outputs,
    last_non_padding_indices,
)  # noqa: E402


def compare_last_token_states(
    output_hidden_states: tuple[torch.Tensor, ...],
    hook_hidden_states: list[torch.Tensor],
    attention_mask: torch.Tensor,
) -> list[dict[str, float | int]]:
    if len(output_hidden_states) != len(hook_hidden_states) + 1:
        raise ValueError(
            "Expected output_hidden_states to include embeddings plus one tensor per hooked block",
        )

    token_indices = last_non_padding_indices(attention_mask)
    batch_positions = torch.arange(attention_mask.shape[0], device=attention_mask.device)
    comparisons = []

    comparable_hook_states = hook_hidden_states[:-1]
    for block_index, hook_layer in enumerate(comparable_hook_states):
        tuple_index = block_index + 1
        old_values = output_hidden_states[tuple_index][batch_positions, token_indices, :]
        hook_values = hook_layer[batch_positions, token_indices, :]
        diff = (old_values - hook_values).float()
        cosine = torch.nn.functional.cosine_similarity(
            old_values.float(),
            hook_values.float(),
            dim=1,
        )
        comparisons.append(
            {
                "hook_block_index": block_index,
                "output_hidden_states_index": tuple_index,
                "max_abs_diff": float(diff.abs().max().item()),
                "mean_abs_diff": float(diff.abs().mean().item()),
                "mean_cosine_similarity": float(cosine.mean().item()),
                "min_cosine_similarity": float(cosine.min().item()),
            },
        )

    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/capture_method_comparison.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(DATASET_NAME, split="train")
    batch_items = dataset.select(range(args.max_examples))
    batch_texts = list(batch_items["sentence"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    all_comparisons: list[list[dict[str, float | int]]] = []
    for start in range(0, len(batch_texts), args.batch_size):
        texts = batch_texts[start : start + args.batch_size]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
            hook_hidden_states = capture_decoder_block_outputs(model, tokens)

        output_hidden_states = outputs.hidden_states
        if output_hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        all_comparisons.append(
            compare_last_token_states(
                output_hidden_states=output_hidden_states,
                hook_hidden_states=hook_hidden_states,
                attention_mask=tokens["attention_mask"],
            ),
        )

    per_layer: list[dict[str, Any]] = []
    for layer_index in range(len(all_comparisons[0])):
        layer_results = [batch[layer_index] for batch in all_comparisons]
        per_layer.append(
            {
                "hook_block_index": layer_results[0]["hook_block_index"],
                "output_hidden_states_index": layer_results[0]["output_hidden_states_index"],
                "max_abs_diff": max(float(result["max_abs_diff"]) for result in layer_results),
                "mean_abs_diff": sum(float(result["mean_abs_diff"]) for result in layer_results)
                / len(layer_results),
                "mean_cosine_similarity": sum(
                    float(result["mean_cosine_similarity"]) for result in layer_results
                )
                / len(layer_results),
                "min_cosine_similarity": min(
                    float(result["min_cosine_similarity"]) for result in layer_results
                ),
            },
        )

    report = {
        "metadata": {
            "dataset": DATASET_NAME,
            "model": MODEL_NAME,
            "num_examples": len(batch_texts),
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "comparison": (
                "forward hook block i vs output_hidden_states[i + 1], excluding the final "
                "decoder block because output_hidden_states[-1] is after final model norm"
            ),
            "num_hook_layers": len(all_comparisons[0]) + 1,
            "num_compared_layers": len(all_comparisons[0]),
        },
        "layers": per_layer,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
