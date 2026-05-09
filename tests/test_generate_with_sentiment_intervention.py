from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "generate_with_sentiment_intervention.py"
    spec = importlib.util.spec_from_file_location(
        "generate_with_sentiment_intervention",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_apply_direction_to_last_token_preserves_tuple_and_modifies_only_last_token() -> None:
    script = load_script()
    hidden = torch.zeros(2, 3, 2)
    output = (hidden, "cache")
    direction = torch.tensor([1.0, -1.0])

    modified_output = script.apply_direction_to_last_token(
        output=output,
        direction=direction,
        alpha=2.0,
    )

    assert isinstance(modified_output, tuple)
    assert modified_output[1] == "cache"
    assert torch.equal(modified_output[0][:, 0, :], torch.zeros(2, 2))
    assert torch.equal(modified_output[0][:, 1, :], torch.zeros(2, 2))
    assert torch.equal(modified_output[0][:, 2, :], torch.tensor([[2.0, -2.0], [2.0, -2.0]]))


def test_apply_direction_to_last_token_alpha_zero_leaves_hidden_states_unchanged() -> None:
    script = load_script()
    hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    direction = torch.tensor([0.25, 0.75])

    modified_output = script.apply_direction_to_last_token(
        output=hidden,
        direction=direction,
        alpha=0.0,
    )

    assert torch.equal(modified_output, hidden)


def test_generate_completion_text_removes_prompt_prefix() -> None:
    script = load_script()

    completion = script.generate_completion_text(
        full_text="Review: The movie felt wonderful and sharp",
        prompt="Review: The movie felt",
    )

    assert completion == " wonderful and sharp"


def test_generate_completion_text_falls_back_to_full_text_without_prefix() -> None:
    script = load_script()

    completion = script.generate_completion_text(
        full_text="A surprising result",
        prompt="Review: The movie felt",
    )

    assert completion == "A surprising result"
