from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "gather_sst2_hidden_states.py"
    spec = importlib.util.spec_from_file_location("gather_sst2_hidden_states", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_indices_is_deterministic_and_disjoint() -> None:
    script = load_script()

    gather_indices, heldout_indices = script.split_indices(size=10, heldout_fraction=0.2, seed=0)
    repeated_gather_indices, repeated_heldout_indices = script.split_indices(
        size=10,
        heldout_fraction=0.2,
        seed=0,
    )

    assert gather_indices == repeated_gather_indices
    assert heldout_indices == repeated_heldout_indices
    assert len(gather_indices) == 8
    assert len(heldout_indices) == 2
    assert set(gather_indices).isdisjoint(heldout_indices)
    assert sorted(gather_indices + heldout_indices) == list(range(10))


def test_last_non_padding_indices() -> None:
    script = load_script()
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
    )

    result = script.last_non_padding_indices(attention_mask)

    assert result.tolist() == [2, 0, 3]
