from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "export_layer_scores.py"
    spec = importlib.util.spec_from_file_location("export_layer_scores", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_layer_scores_exports_train_and_test_scores() -> None:
    script = load_script()
    gathered_data = {
        "hidden_states": torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
        ),
        "labels": torch.tensor([1, 0]),
        "metadata": {"split": "train"},
    }
    directions_data = {
        "directions": torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        ),
        "metadata": {"analysis": "test"},
    }
    validation_data = {
        "scores": torch.tensor([[0.5, -1.0], [-0.25, 2.0]]),
        "labels": torch.tensor([1, 0]),
        "metadata": {"split": "heldout"},
    }

    output = script.build_layer_scores(gathered_data, directions_data, validation_data)

    assert torch.equal(output["train_scores"], torch.tensor([[1.0, 4.0], [5.0, 8.0]]))
    assert torch.equal(output["test_scores"], torch.tensor([[0.5, -1.0], [-0.25, 2.0]]))
    assert torch.equal(output["train_labels"], torch.tensor([1, 0]))
    assert torch.equal(output["test_labels"], torch.tensor([1, 0]))
    assert output["layers"] == [0, 1]
    assert output["metadata"]["score_rule"] == "hidden[layer] dot direction[layer]"


def test_build_layer_scores_requires_matching_test_layer_count() -> None:
    script = load_script()

    with pytest.raises(ValueError, match="same number of layers"):
        script.build_layer_scores(
            gathered_data={
                "hidden_states": torch.ones(2, 2, 3),
                "labels": torch.tensor([0, 1]),
            },
            directions_data={"directions": torch.ones(2, 3)},
            validation_data={
                "scores": torch.ones(2, 3),
                "labels": torch.tensor([0, 1]),
            },
        )
