from __future__ import annotations

import pytest
import torch

from mechinterp.sentiment_probes import (
    ProbeOptions,
    component_directions,
    fit_probe,
    predictions_from_probe_scores,
    score_probe,
)


def separable_hidden_states() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                [[2.0, 0.0], [0.0, 2.0]],
                [[3.0, 0.0], [0.0, 3.0]],
                [[-2.0, 0.0], [0.0, -2.0]],
                [[-3.0, 0.0], [0.0, -3.0]],
            ],
        ),
        torch.tensor([1, 1, 0, 0]),
    )


def test_mean_diff_probe_matches_expected_direction() -> None:
    hidden_states, labels = separable_hidden_states()

    artifact = fit_probe(hidden_states, labels, "mean_diff")

    assert artifact["probe_type"] == "mean_diff"
    assert torch.equal(artifact["directions"], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(
        score_probe(hidden_states, artifact),
        torch.tensor([[2.0, 2.0], [3.0, 3.0], [-2.0, -2.0], [-3.0, -3.0]]),
    )


@pytest.mark.parametrize("probe_type", ["logistic_regression", "linear_svm"])
def test_linear_classifier_probe_separates_simple_classes(probe_type: str) -> None:
    hidden_states, labels = separable_hidden_states()

    artifact = fit_probe(hidden_states, labels, probe_type)  # type: ignore[arg-type]
    scores = score_probe(hidden_states, artifact)
    predictions = predictions_from_probe_scores(scores, artifact)

    assert torch.equal(predictions[:, 0], labels)
    assert torch.equal(predictions[:, 1], labels)
    assert artifact["directions"].shape == (2, 2)
    assert artifact["intercepts"].shape == (2,)


def test_whitened_mean_diff_returns_normalized_directions() -> None:
    hidden_states, labels = separable_hidden_states()

    artifact = fit_probe(
        hidden_states,
        labels,
        "whitened_mean_diff",
        ProbeOptions(whitening_eps=1e-3),
    )

    assert artifact["directions"].shape == (2, 2)
    assert torch.allclose(artifact["directions"].norm(dim=1), torch.ones(2))


def test_pca_deltas_requires_pair_ids() -> None:
    hidden_states, labels = separable_hidden_states()

    with pytest.raises(ValueError, match="pair_ids"):
        fit_probe(hidden_states, labels, "pca_deltas")


def test_pca_deltas_exposes_components() -> None:
    hidden_states = torch.tensor(
        [
            [[2.0, 0.0]],
            [[-2.0, 0.0]],
            [[0.0, 3.0]],
            [[0.0, -3.0]],
        ],
    )
    labels = torch.tensor([1, 0, 1, 0])
    pair_ids = torch.tensor([0, 0, 1, 1])

    artifact = fit_probe(
        hidden_states,
        labels,
        "pca_deltas",
        ProbeOptions(rank=1),
        pair_ids=pair_ids,
    )

    assert artifact["directions"].shape == (1, 2)
    assert component_directions(artifact).shape == (1, 1, 2)


def test_low_rank_subspace_exposes_requested_rank() -> None:
    hidden_states, labels = separable_hidden_states()

    artifact = fit_probe(hidden_states, labels, "low_rank_subspace", ProbeOptions(rank=2))

    assert artifact["directions"].shape == (2, 2)
    assert component_directions(artifact).shape == (2, 2, 2)
