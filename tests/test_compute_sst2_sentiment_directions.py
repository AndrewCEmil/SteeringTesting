from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "compute_sst2_sentiment_directions.py"
    spec = importlib.util.spec_from_file_location("compute_sst2_sentiment_directions", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_directions_subtracts_negative_mean_from_positive_mean() -> None:
    script = load_script()
    hidden_states = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[3.0, 0.0], [0.0, 3.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 2.0], [3.0, 0.0]],
        ],
    )
    labels = torch.tensor([1, 1, 0, 0])

    directions, mean_positive, mean_negative, counts, zero_norm_layers = script.compute_directions(
        hidden_states,
        labels,
    )

    expected_difference = mean_positive - mean_negative
    expected_directions = expected_difference / expected_difference.norm(dim=1, keepdim=True)
    assert directions.shape == (2, 2)
    assert torch.allclose(directions, expected_directions)
    assert torch.allclose(directions.norm(dim=1), torch.ones(2))
    assert counts == {"positive": 2, "negative": 2}
    assert zero_norm_layers == []


def test_compute_directions_requires_both_classes() -> None:
    script = load_script()
    hidden_states = torch.ones(2, 3, 4)
    labels = torch.tensor([1, 1])

    with pytest.raises(ValueError, match="negative"):
        script.compute_directions(hidden_states, labels)


def test_compute_directions_requires_matching_example_count() -> None:
    script = load_script()
    hidden_states = torch.ones(2, 3, 4)
    labels = torch.tensor([0, 1, 1])

    with pytest.raises(ValueError, match="same number of examples"):
        script.compute_directions(hidden_states, labels)


def test_compute_directions_leaves_zero_norm_layers_as_zero() -> None:
    script = load_script()
    hidden_states = torch.tensor(
        [
            [[1.0, 1.0], [3.0, 0.0]],
            [[1.0, 1.0], [0.0, 4.0]],
        ],
    )
    labels = torch.tensor([1, 0])

    directions, _, _, _, zero_norm_layers = script.compute_directions(hidden_states, labels)

    assert torch.equal(directions[0], torch.zeros(2))
    assert torch.allclose(directions[1].norm(), torch.tensor(1.0))
    assert zero_norm_layers == [0]
