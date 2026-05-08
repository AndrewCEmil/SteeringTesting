"""Evaluate combinations of SST-2 sentiment scores across layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def require_tensor(data: dict[str, Any], key: str) -> torch.Tensor:
    value = data.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Expected {key!r} to be a tensor")
    return value


def require_layers(data: dict[str, Any]) -> list[int]:
    layers = data.get("layers")
    if not isinstance(layers, list) or not all(isinstance(layer, int) for layer in layers):
        raise ValueError("Expected 'layers' to be a list of integers")
    return layers


def layer_indices(all_layers: list[int], candidate_layers: list[int]) -> list[int]:
    missing_layers = [layer for layer in candidate_layers if layer not in all_layers]
    if missing_layers:
        raise ValueError(f"Requested layers are missing from input: {missing_layers}")
    return [all_layers.index(layer) for layer in candidate_layers]


def evaluate_scores(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (scores > threshold).astype(np.int64)
    positive_mask = labels == 1
    negative_mask = labels == 0
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "auroc": float(roc_auc_score(labels, scores)),
        "positive_accuracy": float(np.mean(preds[positive_mask] == labels[positive_mask])),
        "negative_accuracy": float(np.mean(preds[negative_mask] == labels[negative_mask])),
        "mean_positive_score": float(np.mean(scores[positive_mask])),
        "mean_negative_score": float(np.mean(scores[negative_mask])),
        "threshold": float(threshold),
    }


def midpoint_threshold(train_scores: np.ndarray, train_labels: np.ndarray) -> float:
    return float(
        0.5 * (train_scores[train_labels == 1].mean() + train_scores[train_labels == 0].mean()),
    )


def wrong_overlap_matrix(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> list[list[float]]:
    wrong = (scores > thresholds.reshape(1, -1)).astype(np.int64) != labels.reshape(-1, 1)
    overlap = np.zeros((scores.shape[1], scores.shape[1]))
    for i in range(scores.shape[1]):
        for j in range(scores.shape[1]):
            overlap[i, j] = np.mean(wrong[:, i] & wrong[:, j])
    return overlap.tolist()


def analyze_layer_scores(
    data: dict[str, Any],
    candidate_layers: list[int],
    input_path: str,
) -> dict[str, Any]:
    all_layers = require_layers(data)
    candidate_indices = layer_indices(all_layers, candidate_layers)

    train_scores = require_tensor(data, "train_scores").float().numpy()
    test_scores = require_tensor(data, "test_scores").float().numpy()
    train_labels = require_tensor(data, "train_labels").numpy()
    test_labels = require_tensor(data, "test_labels").numpy()

    if train_scores.ndim != 2 or test_scores.ndim != 2:
        raise ValueError("train_scores and test_scores must have shape [num_examples, num_layers]")
    if train_labels.ndim != 1 or test_labels.ndim != 1:
        raise ValueError("train_labels and test_labels must have shape [num_examples]")
    if train_scores.shape[0] != train_labels.shape[0]:
        raise ValueError("train scores and labels must have the same number of examples")
    if test_scores.shape[0] != test_labels.shape[0]:
        raise ValueError("test scores and labels must have the same number of examples")
    if len(np.unique(train_labels)) != 2 or len(np.unique(test_labels)) != 2:
        raise ValueError("train and test labels must each include both classes")

    x_train = train_scores[:, candidate_indices]
    x_test = test_scores[:, candidate_indices]

    scaler = StandardScaler()
    x_train_z = scaler.fit_transform(x_train)
    x_test_z = scaler.transform(x_test)

    avg_train = x_train_z.mean(axis=1)
    avg_test = x_test_z.mean(axis=1)
    avg_midpoint = midpoint_threshold(avg_train, train_labels)
    z_average = {
        "threshold_0": evaluate_scores(avg_test, test_labels, threshold=0.0),
        "threshold_midpoint": evaluate_scores(avg_test, test_labels, threshold=avg_midpoint),
    }

    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
    )
    clf.fit(x_train_z, train_labels)
    logit_scores = clf.decision_function(x_test_z)
    logistic_metrics = evaluate_scores(logit_scores, test_labels, threshold=0.0)

    single_layer_metrics: dict[str, dict[str, dict[str, float]]] = {}
    midpoint_thresholds = []
    for layer, idx in zip(candidate_layers, candidate_indices, strict=True):
        train_s = train_scores[:, idx]
        test_s = test_scores[:, idx]
        threshold = midpoint_threshold(train_s, train_labels)
        midpoint_thresholds.append(threshold)
        single_layer_metrics[str(layer)] = {
            "threshold_0": evaluate_scores(test_s, test_labels, threshold=0.0),
            "threshold_midpoint": evaluate_scores(test_s, test_labels, threshold=threshold),
        }

    best_single_layer = max(
        candidate_layers,
        key=lambda layer: single_layer_metrics[str(layer)]["threshold_midpoint"]["accuracy"],
    )
    best_single = single_layer_metrics[str(best_single_layer)]["threshold_midpoint"]
    score_correlation = np.corrcoef(x_test_z.T)

    return {
        "metadata": {
            "input": input_path,
            "candidate_layers": candidate_layers,
            "num_train_examples": int(train_labels.shape[0]),
            "num_test_examples": int(test_labels.shape[0]),
            "normalization": "StandardScaler fit on train scores only",
            "label_rule": "1=positive, 0=negative",
            "prediction_rule": "score > threshold predicts positive",
        },
        "headline": {
            "best_single_layer": int(best_single_layer),
            "best_single_layer_accuracy": best_single["accuracy"],
            "best_single_layer_auroc": best_single["auroc"],
            "z_average_accuracy": z_average["threshold_midpoint"]["accuracy"],
            "z_average_auroc": z_average["threshold_midpoint"]["auroc"],
            "logistic_regression_accuracy": logistic_metrics["accuracy"],
            "logistic_regression_auroc": logistic_metrics["auroc"],
            "logistic_regression_delta_accuracy_vs_best_single": (
                logistic_metrics["accuracy"] - best_single["accuracy"]
            ),
            "logistic_regression_delta_auroc_vs_best_single": (
                logistic_metrics["auroc"] - best_single["auroc"]
            ),
        },
        "z_average": z_average,
        "logistic_regression": {
            "metrics": logistic_metrics,
            "weights_by_layer": {
                str(layer): float(weight)
                for layer, weight in zip(candidate_layers, clf.coef_[0], strict=True)
            },
            "intercept": float(clf.intercept_[0]),
        },
        "single_layer_metrics": single_layer_metrics,
        "score_correlation_matrix": {
            "layers": candidate_layers,
            "values": score_correlation.tolist(),
        },
        "wrong_overlap_matrix": {
            "layers": candidate_layers,
            "values": wrong_overlap_matrix(
                x_test,
                test_labels,
                np.array(midpoint_thresholds),
            ),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    headline = report["headline"]
    layers = report["metadata"]["candidate_layers"]
    weights = report["logistic_regression"]["weights_by_layer"]
    z_metrics = report["z_average"]["threshold_midpoint"]
    logit_metrics = report["logistic_regression"]["metrics"]

    print(f"Layer set: {','.join(str(layer) for layer in layers)}")
    print(
        "Train/test examples: "
        f"{report['metadata']['num_train_examples']}/{report['metadata']['num_test_examples']}",
    )
    print()
    print("Best single layer:")
    print(
        f"  layer {headline['best_single_layer']} midpoint "
        f"accuracy={headline['best_single_layer_accuracy']:.3f} "
        f"auroc={headline['best_single_layer_auroc']:.3f}",
    )
    print()
    print("Combinations:")
    print(
        f"  z-average midpoint accuracy={z_metrics['accuracy']:.3f} "
        f"auroc={z_metrics['auroc']:.3f} "
        f"delta_acc={z_metrics['accuracy'] - headline['best_single_layer_accuracy']:+.3f}",
    )
    print(
        f"  logistic regression accuracy={logit_metrics['accuracy']:.3f} "
        f"auroc={logit_metrics['auroc']:.3f} "
        f"delta_acc={headline['logistic_regression_delta_accuracy_vs_best_single']:+.3f}",
    )
    print()
    print("Logistic weights:")
    for layer in layers:
        print(f"  {layer}: {weights[str(layer)]:+.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument(
        "--output",
        default="outputs/sst2_qwen2_0_5b_layer_combination.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = torch.load(args.input, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError("input file must contain a dictionary")

    report = analyze_layer_scores(data, args.layers, input_path=args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print_summary(report)


if __name__ == "__main__":
    main()
