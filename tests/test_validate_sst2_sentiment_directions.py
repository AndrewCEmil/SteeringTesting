from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "validate_sst2_sentiment_directions.py"
    spec = importlib.util.spec_from_file_location("validate_sst2_sentiment_directions", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_hidden_states_returns_per_example_per_layer_scores() -> None:
    script = load_script()
    hidden_states = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
    )
    directions = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    scores = script.score_hidden_states(hidden_states, directions)

    assert scores.shape == (2, 2)
    assert torch.equal(scores, torch.tensor([[1.0, 4.0], [5.0, 8.0]]))


def test_predictions_from_scores_uses_positive_scores_for_positive_label() -> None:
    script = load_script()
    scores = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [-0.5, 0.5, 2.0],
        ],
    )

    predictions = script.predictions_from_scores(scores)

    assert torch.equal(predictions, torch.tensor([[1, 0, 0], [0, 1, 1]]))


def test_summarize_scores_computes_layer_metrics_and_best_layer() -> None:
    script = load_script()
    scores = torch.tensor(
        [
            [1.0, -1.0],
            [2.0, 3.0],
            [-1.0, -2.0],
            [-2.0, 1.0],
        ],
    )
    labels = torch.tensor([1, 1, 0, 0])

    summary = script.summarize_scores(scores, labels)

    assert summary["num_examples"] == 4
    assert summary["counts"] == {"positive": 2, "negative": 2}
    assert summary["best_layer"] == {"layer": 0, "accuracy": 1.0}
    assert summary["layers"][0]["accuracy"] == 1.0
    assert summary["layers"][0]["positive_accuracy"] == 1.0
    assert summary["layers"][0]["negative_accuracy"] == 1.0
    assert summary["layers"][1]["accuracy"] == 0.5
    assert summary["layers"][1]["positive_accuracy"] == 0.5
    assert summary["layers"][1]["negative_accuracy"] == 0.5
    assert summary["layers"][0]["mean_positive_score"] == 1.5
    assert summary["layers"][0]["mean_negative_score"] == -1.5


def test_score_hidden_states_requires_matching_layer_and_hidden_dimensions() -> None:
    script = load_script()
    hidden_states = torch.ones(2, 3, 4)
    directions = torch.ones(2, 4)

    with pytest.raises(ValueError, match="must match directions"):
        script.score_hidden_states(hidden_states, directions)


def test_summarize_scores_allows_tiny_single_class_samples() -> None:
    script = load_script()
    scores = torch.tensor(
        [
            [-1.0, 2.0],
            [-3.0, 4.0],
        ],
    )
    labels = torch.tensor([0, 0])

    summary = script.summarize_scores(scores, labels)

    assert summary["counts"] == {"positive": 0, "negative": 2}
    assert summary["layers"][0]["accuracy"] == 1.0
    assert summary["layers"][0]["positive_accuracy"] is None
    assert summary["layers"][0]["negative_accuracy"] == 1.0
    assert summary["layers"][0]["mean_positive_score"] is None
    assert summary["layers"][0]["mean_negative_score"] == -2.0
