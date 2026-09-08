"""Built-in sklearn tables and file loading for any local table."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split


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

_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".feather"}


def _require_task_type(task_type: str) -> str:
    if task_type not in {"classification", "regression"}:
        raise ValueError("task_type must be 'classification' or 'regression'")
    return task_type


def load_builtin(name: str) -> DatasetBundle:
    key = name.lower().strip()
    if key not in _LOADERS:
        known = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Unknown built-in dataset '{name}'. Choose one of: {known}, "
            "or pass a table with csv=... and target=..."
        )

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


def _read_table(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix not in _TABLE_SUFFIXES:
        raise ValueError(
            f"Unsupported table '{path}'. Use CSV, TSV, Parquet, or Feather."
        )
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    if suffix == ".feather":
        return pd.read_feather(file_path)
    sep = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(file_path, sep=sep)


def load_table(
    path: str,
    target: str,
    task_type: str,
    *,
    drop: Sequence[str] = (),
) -> DatasetBundle:
    """Load any local table into ``(X, y, task_type)`` for :class:`TabFMDistiller`."""
    task_type = _require_task_type(task_type)
    frame = _read_table(path)
    if target not in frame.columns:
        raise ValueError(f"Target column '{target}' not found in {path}")
    drop_cols = [col for col in drop if col and col != target]
    missing = [col for col in drop_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"drop columns not found in {path}: {missing}")
    y_series = frame[target]
    keep = y_series.notna()
    frame = frame.loc[keep].drop(columns=[target, *drop_cols])
    y = y_series.loc[keep].to_numpy()
    return DatasetBundle(X=frame.reset_index(drop=True), y=y, task_type=task_type, name=path)


def load_csv(
    path: str,
    target: str,
    task_type: str,
    *,
    drop: Sequence[str] = (),
) -> DatasetBundle:
    return load_table(path, target, task_type, drop=drop)


def load_dataset(
    name: str | None = None,
    *,
    csv: str | None = None,
    target: str | None = None,
    task_type: str = "classification",
    drop: Sequence[str] = (),
) -> DatasetBundle:
    if csv:
        if not target:
            raise ValueError("--target is required when loading a table")
        return load_table(csv, target, task_type, drop=drop)
    bundle = load_builtin(name or "breast_cancer")
    if drop:
        missing = [col for col in drop if col not in bundle.X.columns]
        if missing:
            raise ValueError(f"drop columns not found: {missing}")
        bundle = DatasetBundle(
            X=bundle.X.drop(columns=list(drop)),
            y=bundle.y,
            task_type=bundle.task_type,
            name=bundle.name,
        )
    return bundle


def encode_labels(y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary class labels to ``0..K-1`` while keeping the original values."""
    classes, encoded = np.unique(np.asarray(y), return_inverse=True)
    return classes, encoded.astype(int)


def train_eval_split(
    bundle: DatasetBundle,
    *,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Hold out a test split, stratified when every class has at least two rows."""
    stratify = None
    if bundle.task_type == "classification":
        counts = np.unique(bundle.y, return_counts=True)[1]
        if int(counts.min()) >= 2:
            stratify = bundle.y
    return train_test_split(
        bundle.X,
        bundle.y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
