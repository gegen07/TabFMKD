"""Soft-label utilities for Hinton-style knowledge distillation.

TabFM is an in-context learner: its ``fit`` call stores rows as context rather
than updating weights. Soft targets taken from those same rows are therefore
overconfident. This module only shapes already-collected teacher probabilities;
out-of-fold collection lives in :mod:`tabfm_kd.distillation`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _as_2d_probs(probs: NDArray[np.floating]) -> NDArray[np.float64]:
    array = np.asarray(probs, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a (n_samples, n_classes) array, got {array.shape}")
    return np.clip(array, 1e-12, 1.0)


def softmax_from_logits(logits: NDArray[np.floating]) -> NDArray[np.float64]:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def soften_probabilities(
    probs: NDArray[np.floating],
    temperature: float | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Apply Hinton temperature to a probability simplex.

    ``p^(T) = softmax(log(p) / T)``. A scalar ``T`` is broadcast; a vector
    applies a per-sample temperature (adaptive KD).
    """
    probs = _as_2d_probs(probs)
    logits = np.log(probs)
    temps = np.asarray(temperature, dtype=np.float64)
    if temps.ndim == 0:
        if temps <= 0:
            raise ValueError("temperature must be > 0")
        return softmax_from_logits(logits / float(temps))
    if temps.shape != (probs.shape[0],):
        raise ValueError(
            f"Per-sample temperature must have shape {(probs.shape[0],)}, got {temps.shape}"
        )
    if np.any(temps <= 0):
        raise ValueError("all per-sample temperatures must be > 0")
    return softmax_from_logits(logits / temps[:, None])


def predictive_entropy(probs: NDArray[np.floating]) -> NDArray[np.float64]:
    probs = _as_2d_probs(probs)
    return -np.sum(probs * np.log(probs), axis=1)


def adaptive_temperatures(
    probs: NDArray[np.floating],
    base_temperature: float = 3.0,
    beta: float = 1.0,
) -> NDArray[np.float64]:
    """Scale ``T`` by the z-scored teacher entropy of each sample.

    High-entropy (uncertain) rows get a larger temperature so they are not
    oversmoothed relative to confident rows.
    """
    if base_temperature <= 0:
        raise ValueError("base_temperature must be > 0")
    entropy = predictive_entropy(probs)
    scale = entropy.std()
    if scale < 1e-8:
        return np.full(len(entropy), base_temperature, dtype=np.float64)
    z = (entropy - entropy.mean()) / scale
    temps = base_temperature * (1.0 + beta * z)
    return np.clip(temps, 0.25, None)


def confidence_weights(
    probs: NDArray[np.floating],
    sigma: float = 0.15,
) -> NDArray[np.float64]:
    """Bell-shaped weights that peak when the teacher is confident.

    ``w(x) = exp(-(1 - max p(x))^2 / (2 sigma^2))``. Ambiguous rows near the
    decision boundary contribute less to the student update.
    """
    probs = _as_2d_probs(probs)
    confidence = probs.max(axis=1)
    weights = np.exp(-((1.0 - confidence) ** 2) / (2.0 * sigma**2))
    return np.clip(weights, 1e-3, None)


def one_hot(y: NDArray, n_classes: int) -> NDArray[np.float64]:
    labels = np.asarray(y)
    encoded = np.zeros((labels.shape[0], n_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return encoded


def mix_hard_soft(
    soft_probs: NDArray[np.floating],
    y_onehot: NDArray[np.floating],
    alpha: float,
) -> NDArray[np.float64]:
    """``alpha * teacher + (1 - alpha) * hard`` mixed targets.

    ``alpha`` is the soft-label weight from Hinton et al. (2015). The ``T^2``
    gradient correction is implicit for tree students that regress these
    mixed probabilities rather than differentiating a KL term.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    soft = _as_2d_probs(soft_probs)
    hard = np.asarray(y_onehot, dtype=np.float64)
    if soft.shape != hard.shape:
        raise ValueError(f"soft {soft.shape} and hard {hard.shape} shapes must match")
    mixed = alpha * soft + (1.0 - alpha) * hard
    mixed = np.clip(mixed, 1e-12, None)
    return mixed / mixed.sum(axis=1, keepdims=True)


def mix_regression_targets(
    teacher_pred: NDArray[np.floating],
    y_true: NDArray[np.floating],
    alpha: float,
) -> NDArray[np.float64]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return alpha * np.asarray(teacher_pred, dtype=np.float64) + (
        1.0 - alpha
    ) * np.asarray(y_true, dtype=np.float64)
