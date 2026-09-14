"""Evaluation helpers for teacher / student comparison."""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _safe_roc_auc(y_true: NDArray, y_proba: NDArray) -> float | None:
    try:
        if y_proba.shape[1] == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        return None


def _safe_average_precision(y_true: NDArray, y_proba: NDArray) -> float | None:
    """PR-AUC. Binary uses the positive-class score; multiclass is macro OvR."""
    try:
        if y_proba.ndim != 2:
            return None
        if y_proba.shape[1] == 2:
            classes = np.unique(y_true)
            pos_label = classes[1] if len(classes) >= 2 else classes[0]
            return float(
                average_precision_score(y_true, y_proba[:, 1], pos_label=pos_label)
            )
        classes = np.unique(y_true)
        encoded = label_binarize(y_true, classes=classes)
        if encoded.shape[1] == 1:
            return float(average_precision_score(encoded[:, 0], y_proba[:, 0]))
        if encoded.shape[1] != y_proba.shape[1]:
            return None
        return float(average_precision_score(encoded, y_proba, average="macro"))
    except ValueError:
        return None


def gini_from_auc(roc_auc: float) -> float:
    """Somers' D / credit-scoring Gini: ``2 * AUC - 1``."""
    return float(2.0 * roc_auc - 1.0)


def classification_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None:
        metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=np.unique(y_true)))
        auc = _safe_roc_auc(y_true, y_proba)
        if auc is not None:
            metrics["roc_auc"] = auc
            metrics["gini"] = gini_from_auc(auc)
        ap = _safe_average_precision(y_true, y_proba)
        if ap is not None:
            metrics["average_precision"] = ap
    return metrics


def regression_metrics(y_true: NDArray, y_pred: NDArray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def retention(student_score: float, teacher_score: float) -> float:
    if teacher_score == 0:
        return 0.0
    return float(student_score / teacher_score)
