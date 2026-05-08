from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "evaluate_sentiment_intervention.py"
    spec = importlib.util.spec_from_file_location("evaluate_sentiment_intervention", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_direction_to_token_positions_preserves_tuple_and_selected_tokens() -> None:
    script = load_script()
    hidden = torch.zeros(2, 3, 2)
    output = (hidden, "cache")
    token_indices = torch.tensor([1, 2])
    direction = torch.tensor([1.0, -1.0])

    modified_output, original_values, modified_values = script.apply_direction_to_token_positions(
        output=output,
        token_indices=token_indices,
        direction=direction,
        alpha=2.0,
    )

    assert isinstance(modified_output, tuple)
    assert modified_output[1] == "cache"
    assert torch.equal(original_values, torch.zeros(2, 2))
    assert torch.equal(modified_values, torch.tensor([[2.0, -2.0], [2.0, -2.0]]))
    assert torch.equal(modified_output[0][0, 0], torch.tensor([0.0, 0.0]))
    assert torch.equal(modified_output[0][0, 1], torch.tensor([2.0, -2.0]))
    assert torch.equal(modified_output[0][1, 1], torch.tensor([0.0, 0.0]))
    assert torch.equal(modified_output[0][1, 2], torch.tensor([2.0, -2.0]))


def test_apply_direction_alpha_zero_leaves_hidden_states_unchanged() -> None:
    script = load_script()
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    token_indices = torch.tensor([0, 2])
    direction = torch.tensor([0.25, 0.75])

    modified_output, _original_values, _modified_values = script.apply_direction_to_token_positions(
        output=hidden,
        token_indices=token_indices,
        direction=direction,
        alpha=0.0,
    )

    assert torch.equal(modified_output, hidden)


def test_unit_direction_score_delta_equals_alpha() -> None:
    script = load_script()
    hidden = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    token_indices = torch.tensor([0, 0])
    direction = torch.tensor([0.6, 0.8])
    alpha = 1.5

    _modified_output, original_values, modified_values = script.apply_direction_to_token_positions(
        output=hidden,
        token_indices=token_indices,
        direction=direction,
        alpha=alpha,
    )
    baseline_scores = (original_values * direction.unsqueeze(0)).sum(dim=1)
    intervened_scores = (modified_values * direction.unsqueeze(0)).sum(dim=1)

    assert torch.allclose(intervened_scores - baseline_scores, torch.full((2,), alpha))


def test_summarize_alpha_reports_flips_and_delta_error() -> None:
    script = load_script()
    baseline_scores = torch.tensor([-0.25, 0.25, -2.0, 2.0])
    intervened_scores = torch.tensor([0.75, 1.25, -1.0, 3.0])
    labels = torch.tensor([0, 1, 0, 1])
    baseline_predictions = script.predictions_from_scores(baseline_scores)

    summary = script.summarize_alpha(
        alpha=1.0,
        baseline_scores=baseline_scores,
        intervened_scores=intervened_scores,
        baseline_predictions=baseline_predictions,
        labels=labels,
    )

    assert summary["alpha"] == 1.0
    assert summary["mean_score_delta"] == 1.0
    assert summary["mean_abs_delta_error"] == 0.0
    assert summary["max_abs_delta_error"] == 0.0
    assert summary["flip_rate"] == 0.25
    assert summary["accuracy"] == 0.75
    assert summary["positive_accuracy"] == 1.0
    assert summary["negative_accuracy"] == 0.5
