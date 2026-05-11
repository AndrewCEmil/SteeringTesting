"""Per-layer sentiment probes for hidden-state analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.svm import LinearSVC  # type: ignore[import-untyped]

ProbeType = Literal[
    "mean_diff",
    "logistic_regression",
    "linear_svm",
    "whitened_mean_diff",
    "pca_deltas",
    "low_rank_subspace",
]


@dataclass(frozen=True)
class ProbeOptions:
    rank: int = 1
    c: float = 1.0
    max_iter: int = 1000
    whitening_eps: float = 1e-4


def validate_hidden_states_and_labels(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [num_examples, num_layers, hidden_size]")
    if labels.dim() != 1:
        raise ValueError("labels must have shape [num_examples]")
    if hidden_states.shape[0] != labels.shape[0]:
        raise ValueError("hidden_states and labels must have the same number of examples")

    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_count = int(positive_mask.sum().item())
    negative_count = int(negative_mask.sum().item())
    if positive_count == 0:
        raise ValueError("labels must include at least one positive example")
    if negative_count == 0:
        raise ValueError("labels must include at least one negative example")

    return positive_mask, negative_mask, {"positive": positive_count, "negative": negative_count}


def normalize_rows(vectors: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    norms = vectors.norm(dim=1, keepdim=True)
    nonzero_norms = norms.squeeze(dim=1) != 0
    normalized = torch.zeros_like(vectors)
    normalized[nonzero_norms] = vectors[nonzero_norms] / norms[nonzero_norms]
    zero_norm_layers = torch.nonzero(~nonzero_norms).squeeze(dim=1).tolist()
    return normalized, zero_norm_layers


def _normalize_matrix_rows(matrix: torch.Tensor) -> torch.Tensor:
    norms = matrix.norm(dim=-1, keepdim=True)
    return torch.where(norms > 0, matrix / norms.clamp_min(torch.finfo(matrix.dtype).eps), matrix)


def _as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().float().numpy()


def _orient_positive(
    direction: torch.Tensor,
    positive_hidden: torch.Tensor,
    negative_hidden: torch.Tensor,
) -> torch.Tensor:
    positive_score = torch.matmul(positive_hidden, direction).mean()
    negative_score = torch.matmul(negative_hidden, direction).mean()
    if positive_score < negative_score:
        return -direction
    return direction


def fit_probe(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    probe_type: ProbeType,
    options: ProbeOptions | None = None,
    pair_ids: torch.Tensor | None = None,
) -> dict[str, Any]:
    options = options or ProbeOptions()
    positive_mask, negative_mask, counts = validate_hidden_states_and_labels(hidden_states, labels)

    if probe_type == "mean_diff":
        return fit_mean_diff(hidden_states, positive_mask, negative_mask, counts)
    if probe_type == "logistic_regression":
        return fit_linear_classifier(hidden_states, labels, counts, probe_type, options)
    if probe_type == "linear_svm":
        return fit_linear_classifier(hidden_states, labels, counts, probe_type, options)
    if probe_type == "whitened_mean_diff":
        return fit_whitened_mean_diff(hidden_states, positive_mask, negative_mask, counts, options)
    if probe_type == "pca_deltas":
        return fit_pca_deltas(hidden_states, labels, counts, options, pair_ids)
    if probe_type == "low_rank_subspace":
        return fit_low_rank_subspace(
            hidden_states,
            labels,
            positive_mask,
            negative_mask,
            counts,
            options,
        )
    raise ValueError(f"Unknown probe type: {probe_type}")


def fit_mean_diff(
    hidden_states: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    counts: dict[str, int],
) -> dict[str, Any]:
    mean_positive = hidden_states[positive_mask].mean(dim=0)
    mean_negative = hidden_states[negative_mask].mean(dim=0)
    directions, zero_norm_layers = normalize_rows(mean_positive - mean_negative)
    return {
        "probe_type": "mean_diff",
        "directions": directions,
        "mean_positive": mean_positive,
        "mean_negative": mean_negative,
        "counts": counts,
        "thresholds": torch.zeros(directions.shape[0]),
        "metadata": {
            "analysis": "mean_positive_minus_mean_negative",
            "normalized": True,
            "zero_norm_layers": zero_norm_layers,
            "score_rule": "hidden[layer] dot direction[layer]",
            "prediction_rule": "score > 0 predicts positive",
        },
    }


def fit_linear_classifier(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    counts: dict[str, int],
    probe_type: Literal["logistic_regression", "linear_svm"],
    options: ProbeOptions,
) -> dict[str, Any]:
    directions = []
    intercepts = []
    for layer in range(hidden_states.shape[1]):
        x = _as_numpy(hidden_states[:, layer, :])
        y = _as_numpy(labels).astype(np.int64)
        if probe_type == "logistic_regression":
            clf = LogisticRegression(
                C=options.c,
                class_weight="balanced",
                max_iter=options.max_iter,
                random_state=0,
            )
        else:
            clf = LinearSVC(
                C=options.c,
                class_weight="balanced",
                max_iter=options.max_iter,
                random_state=0,
                dual="auto",
            )
        clf.fit(x, y)
        directions.append(torch.from_numpy(clf.coef_[0]).to(hidden_states.dtype))
        intercepts.append(float(clf.intercept_[0]))

    raw_directions = torch.stack(directions, dim=0)
    normalized, zero_norm_layers = normalize_rows(raw_directions)
    return {
        "probe_type": probe_type,
        "directions": normalized,
        "raw_directions": raw_directions,
        "intercepts": torch.tensor(intercepts, dtype=hidden_states.dtype),
        "counts": counts,
        "thresholds": torch.zeros(hidden_states.shape[1]),
        "metadata": {
            "analysis": probe_type,
            "normalized": True,
            "zero_norm_layers": zero_norm_layers,
            "score_rule": "hidden[layer] dot raw_direction[layer] + intercept[layer]",
            "prediction_rule": "score > 0 predicts positive",
            "c": options.c,
            "max_iter": options.max_iter,
        },
    }


def fit_whitened_mean_diff(
    hidden_states: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    counts: dict[str, int],
    options: ProbeOptions,
) -> dict[str, Any]:
    mean_positive = hidden_states[positive_mask].mean(dim=0)
    mean_negative = hidden_states[negative_mask].mean(dim=0)
    centered = hidden_states - hidden_states.mean(dim=0, keepdim=True)
    directions = []
    for layer in range(hidden_states.shape[1]):
        x = centered[:, layer, :].float()
        cov = x.T @ x / max(x.shape[0] - 1, 1)
        eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
        solved = torch.linalg.solve(
            cov + options.whitening_eps * eye,
            mean_positive[layer] - mean_negative[layer],
        )
        directions.append(solved.to(hidden_states.dtype))
    raw_directions = torch.stack(directions, dim=0)
    normalized, zero_norm_layers = normalize_rows(raw_directions)
    return {
        "probe_type": "whitened_mean_diff",
        "directions": normalized,
        "raw_directions": raw_directions,
        "mean_positive": mean_positive,
        "mean_negative": mean_negative,
        "counts": counts,
        "thresholds": torch.zeros(hidden_states.shape[1]),
        "metadata": {
            "analysis": "inverse_covariance_mean_positive_minus_mean_negative",
            "normalized": True,
            "zero_norm_layers": zero_norm_layers,
            "score_rule": "hidden[layer] dot raw_direction[layer]",
            "prediction_rule": "score > 0 predicts positive",
            "whitening_eps": options.whitening_eps,
        },
    }


def paired_deltas(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    pair_ids: torch.Tensor | None,
) -> torch.Tensor:
    if pair_ids is None:
        raise ValueError("pair_ids are required for pca_deltas")
    if pair_ids.dim() != 1 or pair_ids.shape[0] != labels.shape[0]:
        raise ValueError("pair_ids must have shape [num_examples]")

    deltas = []
    for pair_id in torch.unique(pair_ids).tolist():
        mask = pair_ids == pair_id
        pair_labels = labels[mask]
        if int((pair_labels == 1).sum().item()) != 1 or int((pair_labels == 0).sum().item()) != 1:
            raise ValueError(
                "each pair_id must identify exactly one positive and one negative example",
            )
        positive = hidden_states[mask][pair_labels == 1][0]
        negative = hidden_states[mask][pair_labels == 0][0]
        deltas.append(positive - negative)
    if not deltas:
        raise ValueError("pair_ids did not identify any pairs")
    return torch.stack(deltas, dim=0)


def fit_pca_deltas(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    counts: dict[str, int],
    options: ProbeOptions,
    pair_ids: torch.Tensor | None,
) -> dict[str, Any]:
    deltas = paired_deltas(hidden_states, labels, pair_ids)
    return fit_components_from_deltas(
        deltas=deltas,
        hidden_states=hidden_states,
        labels=labels,
        counts=counts,
        probe_type="pca_deltas",
        options=options,
    )


def fit_low_rank_subspace(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    counts: dict[str, int],
    options: ProbeOptions,
) -> dict[str, Any]:
    positive_centered = hidden_states[positive_mask] - hidden_states[positive_mask].mean(
        dim=0,
        keepdim=True,
    )
    negative_centered = hidden_states[negative_mask] - hidden_states[negative_mask].mean(
        dim=0,
        keepdim=True,
    )
    deltas = torch.cat([positive_centered, -negative_centered], dim=0)
    mean_delta = hidden_states[positive_mask].mean(dim=0) - hidden_states[negative_mask].mean(dim=0)
    deltas = torch.cat([mean_delta.unsqueeze(0), deltas], dim=0)
    return fit_components_from_deltas(
        deltas=deltas,
        hidden_states=hidden_states,
        labels=labels,
        counts=counts,
        probe_type="low_rank_subspace",
        options=options,
    )


def fit_components_from_deltas(
    deltas: torch.Tensor,
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    counts: dict[str, int],
    probe_type: Literal["pca_deltas", "low_rank_subspace"],
    options: ProbeOptions,
) -> dict[str, Any]:
    rank = min(options.rank, deltas.shape[0], deltas.shape[2])
    if rank < 1:
        raise ValueError("rank must be at least 1")

    per_layer_components = []
    explained_variance = []
    for layer in range(deltas.shape[1]):
        pca = PCA(n_components=rank, random_state=0)
        components_np = pca.fit(_as_numpy(deltas[:, layer, :])).components_
        components = torch.from_numpy(components_np).to(hidden_states.dtype)
        components[0] = _orient_positive(
            components[0],
            hidden_states[labels == 1, layer, :],
            hidden_states[labels == 0, layer, :],
        )
        per_layer_components.append(components)
        explained_variance.append(
            torch.from_numpy(pca.explained_variance_ratio_).to(hidden_states.dtype)
        )

    components_tensor = _normalize_matrix_rows(torch.stack(per_layer_components, dim=0))
    directions = components_tensor[:, 0, :]
    return {
        "probe_type": probe_type,
        "directions": directions,
        "components": components_tensor,
        "explained_variance_ratio": torch.stack(explained_variance, dim=0),
        "counts": counts,
        "thresholds": torch.zeros(hidden_states.shape[1]),
        "metadata": {
            "analysis": probe_type,
            "normalized": True,
            "rank": rank,
            "score_rule": "hidden[layer] dot first_component[layer]",
            "prediction_rule": "score > 0 predicts positive",
        },
    }


def score_probe(hidden_states: torch.Tensor, artifact: dict[str, Any]) -> torch.Tensor:
    if hidden_states.dim() != 3:
        raise ValueError("hidden_states must have shape [num_examples, num_layers, hidden_size]")

    probe_type = artifact.get("probe_type") or artifact.get("metadata", {}).get("probe_type")
    has_raw_directions = isinstance(artifact.get("raw_directions"), torch.Tensor)
    if probe_type in {"logistic_regression", "linear_svm"} and has_raw_directions:
        directions = artifact["raw_directions"]
        intercepts = artifact.get("intercepts")
    elif probe_type == "whitened_mean_diff" and has_raw_directions:
        directions = artifact["raw_directions"]
        intercepts = None
    else:
        directions = artifact.get("directions")
        intercepts = None

    if not isinstance(directions, torch.Tensor):
        raise ValueError("probe artifact must contain tensor directions")
    if directions.dim() != 2:
        raise ValueError("directions must have shape [num_layers, hidden_size]")
    if hidden_states.shape[1:] != directions.shape:
        raise ValueError("hidden_states layer/hidden dimensions must match directions")

    scores = (hidden_states * directions.unsqueeze(0).to(hidden_states.device)).sum(dim=2)
    if intercepts is not None:
        if not isinstance(intercepts, torch.Tensor) or intercepts.shape != (directions.shape[0],):
            raise ValueError("intercepts must have shape [num_layers]")
        scores = scores + intercepts.to(scores.device).unsqueeze(0)
    return scores


def predictions_from_probe_scores(
    scores: torch.Tensor,
    artifact: dict[str, Any] | None = None,
) -> torch.Tensor:
    thresholds = artifact.get("thresholds") if artifact is not None else None
    if isinstance(thresholds, torch.Tensor):
        return (scores > thresholds.to(scores.device).unsqueeze(0)).to(torch.long)
    return (scores > 0).to(torch.long)


def primary_directions(artifact: dict[str, Any]) -> torch.Tensor:
    directions = artifact.get("directions")
    if not isinstance(directions, torch.Tensor):
        raise ValueError("probe artifact must contain tensor directions")
    if directions.dim() != 2:
        raise ValueError("directions must have shape [num_layers, hidden_size]")
    return directions


def component_directions(artifact: dict[str, Any]) -> torch.Tensor:
    components = artifact.get("components")
    if not isinstance(components, torch.Tensor):
        raise ValueError("probe artifact does not contain multi-vector components")
    if components.dim() != 3:
        raise ValueError("components must have shape [num_layers, rank, hidden_size]")
    return components
