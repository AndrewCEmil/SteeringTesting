from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch


def load_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "combine_layer_scores.py"
    spec = importlib.util.spec_from_file_location("combine_layer_scores", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_layer_scores_reports_headline_and_matrices() -> None:
    script = load_script()
    data = {
        "train_scores": torch.tensor(
            [
                [2.0, 1.0, 0.0],
                [3.0, 1.5, 0.0],
                [-2.0, -1.0, 0.0],
                [-3.0, -1.5, 0.0],
            ],
        ),
        "test_scores": torch.tensor(
            [
                [2.5, 1.2, 0.0],
                [1.5, 0.8, 0.0],
                [-2.5, -1.2, 0.0],
                [-1.5, -0.8, 0.0],
            ],
        ),
        "train_labels": torch.tensor([1, 1, 0, 0]),
        "test_labels": torch.tensor([1, 1, 0, 0]),
        "layers": [0, 1, 2],
    }

    report = script.analyze_layer_scores(data, candidate_layers=[0, 1], input_path="scores.pt")

    assert report["metadata"]["input"] == "scores.pt"
    assert report["metadata"]["candidate_layers"] == [0, 1]
    assert report["headline"]["best_single_layer"] in {0, 1}
    assert report["headline"]["best_single_layer_accuracy"] == 1.0
    assert report["headline"]["z_average_accuracy"] == 1.0
    assert report["headline"]["logistic_regression_accuracy"] == 1.0
    assert list(report["logistic_regression"]["weights_by_layer"]) == ["0", "1"]
    assert report["score_correlation_matrix"]["layers"] == [0, 1]
    assert report["wrong_overlap_matrix"]["layers"] == [0, 1]
    assert len(report["score_correlation_matrix"]["values"]) == 2
    assert len(report["wrong_overlap_matrix"]["values"]) == 2


def test_analyze_layer_scores_rejects_missing_candidate_layers() -> None:
    script = load_script()
    data = {
        "train_scores": torch.ones(4, 2),
        "test_scores": torch.ones(4, 2),
        "train_labels": torch.tensor([1, 1, 0, 0]),
        "test_labels": torch.tensor([1, 1, 0, 0]),
        "layers": [0, 1],
    }

    with pytest.raises(ValueError, match="missing"):
        script.analyze_layer_scores(data, candidate_layers=[2], input_path="scores.pt")


def test_analyze_layer_scores_accepts_bfloat16_scores() -> None:
    script = load_script()
    data = {
        "train_scores": torch.tensor(
            [
                [2.0, 1.0],
                [3.0, 1.5],
                [-2.0, -1.0],
                [-3.0, -1.5],
            ],
            dtype=torch.bfloat16,
        ),
        "test_scores": torch.tensor(
            [
                [2.5, 1.2],
                [1.5, 0.8],
                [-2.5, -1.2],
                [-1.5, -0.8],
            ],
            dtype=torch.bfloat16,
        ),
        "train_labels": torch.tensor([1, 1, 0, 0]),
        "test_labels": torch.tensor([1, 1, 0, 0]),
        "layers": [0, 1],
    }

    report = script.analyze_layer_scores(data, candidate_layers=[0, 1], input_path="scores.pt")

    assert report["headline"]["logistic_regression_accuracy"] == 1.0
