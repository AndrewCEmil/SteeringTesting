from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


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


class TupleLayer(nn.Module):
    def __init__(self, add_value: float) -> None:
        super().__init__()
        self.add_value = add_value

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden_states + self.add_value,)


class DummyInnerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TupleLayer(1.0), TupleLayer(2.0)])


class DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = DummyInnerModel()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = input_ids.float().unsqueeze(dim=-1)
        for layer in self.model.layers:
            layer_output = layer(hidden_states)
            hidden_states = layer_output[0]
        return hidden_states


def test_capture_decoder_block_outputs_captures_tuple_hidden_states() -> None:
    script = load_script()
    model = DummyModel()
    tokens = {"input_ids": torch.tensor([[1, 2], [3, 4]])}

    captures = script.capture_decoder_block_outputs(model, tokens)

    assert len(captures) == 2
    assert torch.equal(captures[0], torch.tensor([[[2.0], [3.0]], [[4.0], [5.0]]]))
    assert torch.equal(captures[1], torch.tensor([[[4.0], [5.0]], [[6.0], [7.0]]]))
