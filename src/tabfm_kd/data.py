"""Built-in sklearn tables and CSV loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine


@dataclass(frozen=True)
class DatasetBundle:
    X: pd.DataFrame
    y: np.ndarray
    task_type: str
    name: str


_LOADERS = {
    "breast_cancer": (load_breast_cancer, "classification"),
    "wine": (load_wine, "classification"),
    "iris": (load_iris, "classification"),
    "diabetes": (load_diabetes, "regression"),
}


def load_builtin(name: str) -> DatasetBundle:
    key = name.lower().strip()

    loader, task_type = _LOADERS[key]
    bunch = loader(as_frame=True)
    frame = bunch.frame
    target_name = bunch.target.name if hasattr(bunch.target, "name") else "target"
    if target_name in frame.columns:
        X = frame.drop(columns=[target_name])
        y = frame[target_name].to_numpy()
    else:
        X = pd.DataFrame(bunch.data)
        y = np.asarray(bunch.target)
    return DatasetBundle(X=X, y=y, task_type=task_type, name=key)


def load_csv(path: str, target: str, task_type: str) -> DatasetBundle:
    frame = pd.read_csv(path)
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    X = frame.drop(columns=[target])
    y = frame[target].to_numpy()
    return DatasetBundle(X=X, y=y, task_type=task_type, name=path)


def load_dataset(
    name: str | None = None,
    *,
    csv: str | None = None,
    target: str | None = None,
    task_type: str = "classification",
) -> DatasetBundle:
    if csv:
        if not target:
            raise ValueError("--target is required when loading a CSV")
        return load_csv(csv, target, task_type)
    return load_builtin(name or "breast_cancer")


def encode_labels(y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary class labels to ``0..K-1`` while keeping the original values."""
    classes, encoded = np.unique(np.asarray(y), return_inverse=True)
    return classes, encoded.astype(int)
