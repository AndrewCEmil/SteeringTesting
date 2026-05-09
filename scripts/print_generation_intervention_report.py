"""Print a compact grouped view of generation intervention results."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def group_results(results: list[dict[str, Any]]) -> dict[str, dict[float, list[str]]]:
    grouped: dict[str, dict[float, list[str]]] = defaultdict(lambda: defaultdict(list))
    for result in results:
        prompt = str(result["prompt"])
        alpha = float(result["alpha"])
        grouped[prompt][alpha].append(str(result["completion"]))
    return grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/smoke_sentiment_generation_layer14.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input).read_text())
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("input JSON must contain a 'results' list")

    grouped = group_results(results)
    for prompt, alpha_groups in grouped.items():
        print(f"Prompt: {prompt}")
        print()
        for alpha in sorted(alpha_groups):
            print(f"alpha {alpha:g}:")
            for completion in alpha_groups[alpha]:
                print(f"  - {completion}")
            print()


if __name__ == "__main__":
    main()
