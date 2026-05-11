"""Generate short continuations with sentiment-direction steering."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from abliteration.sentiment_probes import component_directions, primary_directions

sys.path.append(str(Path(__file__).parent))
from gather_sst2_hidden_states import (  # noqa: E402
    MODEL_NAME,
    decoder_layers,
    layer_hidden_from_output,
)

DEFAULT_PROMPTS = [
    "Review: The restaurant was",
    "Review: The movie felt",
    "I walked out of the theater feeling",
    "The product experience was",
]


def apply_direction_to_last_token(output: Any, direction: torch.Tensor, alpha: float) -> Any:
    hidden = layer_hidden_from_output(output)
    modified_hidden = hidden.clone()
    modified_hidden[:, -1, :] += alpha * direction
    if isinstance(output, tuple):
        return (modified_hidden, *output[1:])
    return modified_hidden


def generate_completion_text(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :]
    return full_text


def generate_for_alpha(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    direction: torch.Tensor,
    hook_block_index: int,
    alpha: float,
    max_new_tokens: int,
    num_samples: int,
    temperature: float,
    top_p: float,
    device: str,
) -> list[dict[str, Any]]:
    layers = decoder_layers(model)
    if hook_block_index < 0 or hook_block_index >= len(layers):
        raise ValueError(f"hook_block_index must be between 0 and {len(layers) - 1}")

    def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
        return apply_direction_to_last_token(output=output, direction=direction, alpha=alpha)

    tokens = tokenizer(prompt, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}

    handle = layers[hook_block_index].register_forward_hook(hook)
    try:
        with torch.no_grad():
            generated = model.generate(
                **tokens,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_samples,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
    finally:
        handle.remove()

    records = []
    for sample_index, output_ids in enumerate(generated):
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        records.append(
            {
                "prompt": prompt,
                "alpha": float(alpha),
                "sample_index": sample_index,
                "text": text,
                "completion": generate_completion_text(text, prompt),
            },
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directions", default="outputs/smoke_sentiment_directions.pt")
    parser.add_argument("--output", default="outputs/smoke_sentiment_generation_layer14.json")
    parser.add_argument("--hook-block-index", type=int, default=13)
    parser.add_argument("--direction-layer-index", type=int, default=14)
    parser.add_argument("--component-index", type=int, default=None)
    parser.add_argument("--alphas", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    results = []
    for prompt in args.prompts:
        for alpha in args.alphas:
            results.extend(
                generate_for_alpha(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    direction=direction,
                    hook_block_index=args.hook_block_index,
                    alpha=alpha,
                    max_new_tokens=args.max_new_tokens,
                    num_samples=args.num_samples,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                ),
            )

    report = {
        "metadata": {
            "model": MODEL_NAME,
            "directions": args.directions,
            "hook_block_index": args.hook_block_index,
            "direction_layer_index": args.direction_layer_index,
            "component_index": args.component_index,
            "alphas": [float(alpha) for alpha in args.alphas],
            "max_new_tokens": args.max_new_tokens,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "prompts": args.prompts,
            "intervention_rule": (
                "add alpha * direction to hidden[:, -1, :] during every generation forward pass"
            ),
        },
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
