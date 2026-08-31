"""Orchestrate TabFM → XGBoost knowledge distillation.

The important design choice is k-fold cross-prediction. TabFM conditions on
whatever rows ``fit`` stored as context. If those rows are also the rows we
ask it to label, the soft targets collapse toward one-hot memorization
("ICL identity leakage"). Each sample is therefore labeled by a teacher that
never saw it in context.

Each fold teacher is also fit on a stratified sample of the complementary
folds rather than the full split. TabFM only consumes a small in-context
window, so materializing a huge train fold would OOM without helping quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import KFold, StratifiedKFold

from tabfm_kd.losses import (
    adaptive_temperatures,
    confidence_weights,
    soften_probabilities,
)
from tabfm_kd.metrics import classification_metrics, regression_metrics, retention
from tabfm_kd.student import XGBoostStudent
from tabfm_kd.teacher import Teacher, build_teacher


def _as_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    array = np.asarray(X)
    return pd.DataFrame(array, columns=[f"f{i}" for i in range(array.shape[1])])


def _rows(X: pd.DataFrame, idx: NDArray[np.intp]) -> pd.DataFrame:
    return X.iloc[idx]


def _align_proba(
    proba: NDArray[np.floating],
    src_classes: NDArray,
    dst_classes: NDArray,
) -> NDArray[np.float64]:
    aligned = np.zeros((len(proba), len(dst_classes)), dtype=np.float64)
    src_index = {label: i for i, label in enumerate(src_classes)}
    for dest_j, label in enumerate(dst_classes):
        src_i = src_index.get(label)
        if src_i is not None:
            aligned[:, dest_j] = proba[:, src_i]
    totals = aligned.sum(axis=1, keepdims=True)
    missing = totals[:, 0] < 1e-12
    if np.any(missing):
        aligned[missing] = 1.0 / len(dst_classes)
        totals = aligned.sum(axis=1, keepdims=True)
    return aligned / np.clip(totals, 1e-12, None)


def _subsample_indices(
    y: NDArray,
    idx: NDArray[np.intp],
    sample_size: int,
    task_type: str,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    """Pick ``sample_size`` rows from ``idx``, stratified when classifying."""
    idx = np.asarray(idx, dtype=np.intp)
    n = len(idx)
    if sample_size <= 0 or sample_size >= n:
        return idx
    if task_type != "classification":
        chosen = rng.choice(n, size=sample_size, replace=False)
        return np.sort(idx[chosen])

    y_fold = y[idx]
    classes, counts = np.unique(y_fold, return_counts=True)
    selected: list[NDArray[np.intp]] = []
    remaining = sample_size
    n_classes = len(classes)
    for i, (cls, count) in enumerate(zip(classes, counts)):
        local = np.flatnonzero(y_fold == cls)
        rng.shuffle(local)
        if i == n_classes - 1:
            take = min(remaining, int(count))
        else:
            proportional = int(round(sample_size * int(count) / n))
            take = min(max(1, proportional), int(count), remaining)
        remaining -= take
        if take > 0:
            selected.append(idx[local[:take]])
    if remaining > 0:
        used = np.concatenate(selected) if selected else np.empty(0, dtype=np.intp)
        unused_mask = np.isin(idx, used, invert=True)
        unused = idx[unused_mask]
        extra = min(remaining, len(unused))
        if extra:
            pick = rng.choice(len(unused), size=extra, replace=False)
            selected.append(unused[pick])
    if not selected:
        chosen = rng.choice(n, size=sample_size, replace=False)
        return np.sort(idx[chosen])
    return np.sort(np.concatenate(selected))


def _context_indices(
    y: NDArray,
    idx: NDArray[np.intp],
    *,
    sample_size: int | None,
    task_type: str,
    random_state: int,
) -> NDArray[np.intp]:
    if sample_size is None:
        return np.asarray(idx, dtype=np.intp)
    rng = np.random.default_rng(random_state)
    return _subsample_indices(y, idx, sample_size, task_type, rng)


def _resolve_teacher_sample_size(
    sample_size: int | None,
    *,
    max_num_rows: int,
    n_estimators: int,
) -> int | None:
    """``None`` → TabFM-sized pool; ``0`` → full fold; ``>0`` → that many rows."""
    if sample_size == 0:
        return None
    if sample_size is not None:
        if sample_size < 0:
            raise ValueError("teacher_sample_size must be >= 0")
        return sample_size
    return max(1, int(max_num_rows) * max(int(n_estimators), 1))


def collect_oof_targets(
    teacher: Teacher,
    X: pd.DataFrame,
    y: NDArray,
    *,
    task_type: str,
    n_folds: int,
    random_state: int,
    classes: NDArray | None = None,
    sample_size: int | None = None,
) -> NDArray[np.float64]:
    """Out-of-fold teacher predictions used as distillation targets.

    Each fold teacher is fit on a sample of the complementary folds
    (``sample_size`` rows; ``None`` keeps the full split) and then labels
    the held-out rows. Sampling never pulls from the validation fold, so
    ICL identity leakage stays blocked.
    """
    n_samples = len(y)
    n_folds = _effective_folds(y, n_folds, task_type)
    if task_type == "classification":
        if classes is None:
            classes = np.unique(y)
        soft = np.zeros((n_samples, len(classes)), dtype=np.float64)
        splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=random_state
        )
        splits = splitter.split(X, y)
    else:
        soft = np.zeros(n_samples, dtype=np.float64)
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = splitter.split(X)

    for fold_i, (train_idx, valid_idx) in enumerate(splits):
        context_idx = _context_indices(
            y,
            np.asarray(train_idx, dtype=np.intp),
            sample_size=sample_size,
            task_type=task_type,
            random_state=random_state + fold_i + 1,
        )
        fold_teacher = teacher.clone_unfitted()
        fold_teacher.fit(_rows(X, context_idx), y[context_idx])
        if task_type == "classification":
            proba = np.asarray(fold_teacher.predict_proba(_rows(X, valid_idx)))
            fold_classes = getattr(fold_teacher, "classes_", classes)
            soft[valid_idx] = _align_proba(proba, np.asarray(fold_classes), classes)
        else:
            soft[valid_idx] = np.asarray(
                fold_teacher.predict(_rows(X, valid_idx))
            ).reshape(-1)
    return soft


def _effective_folds(y: NDArray, n_folds: int, task_type: str) -> int:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if task_type != "classification":
        return min(n_folds, len(y))
    counts = np.unique(y, return_counts=True)[1]
    return max(2, min(n_folds, int(counts.min())))


@dataclass
class DistillConfig:
    """Knobs for TabFM → XGBoost distillation.

    ``teacher_sample_size`` caps how many rows each teacher ``fit`` sees.
    ``None`` (default) uses ``teacher_max_num_rows * teacher_n_estimators`` —
    enough for diverse TabFM bags without copying a huge fold into memory.
    Pass ``0`` to disable sampling and fit on the full fold.
    """

    task_type: str = "classification"
    teacher: str = "tabfm"
    teacher_backend: str = "pytorch"
    teacher_n_estimators: int = 8
    teacher_max_num_rows: int = 100
    teacher_sample_size: int | None = None
    temperature: float = 3.0
    alpha: float = 0.7
    n_folds: int = 5
    adaptive_temperature: bool = True
    confidence_weighting: bool = True
    augment_factor: float | None = None
    xgb_params: dict[str, Any] = field(default_factory=dict)
    teacher_params: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42


class TabFMDistiller:
    """Compress a tabular foundation-model teacher into an XGBoost student.

    After :meth:`fit`, :meth:`predict` / :meth:`predict_proba` hit only the
    student. :meth:`save` drops the teacher so the artifact has no TabFM
    dependency at serve time.
    """

    def __init__(
        self,
        config: DistillConfig | None = None,
        teacher: Teacher | None = None,
        student: XGBoostStudent | None = None,
    ) -> None:
        self.config = config or DistillConfig()
        self.teacher = teacher or build_teacher(
            self.config.teacher,
            task_type=self.config.task_type,
            backend=self.config.teacher_backend,
            n_estimators=self.config.teacher_n_estimators,
            max_num_rows=self.config.teacher_max_num_rows,
            random_state=self.config.random_state,
            **self.config.teacher_params,
        )
        self.student = student or XGBoostStudent(
            task_type=self.config.task_type,
            alpha=self.config.alpha,
            random_state=self.config.random_state,
            **self.config.xgb_params,
        )
        self.soft_targets_: NDArray[np.float64] | None = None
        self.sample_weight_: NDArray[np.float64] | None = None
        self.classes_: NDArray | None = None
        self.teacher_fitted_: bool = False
        self._train_X_: pd.DataFrame | None = None
        self._train_y_: NDArray | None = None

    def fit(self, X: Any, y: Any) -> TabFMDistiller:
        frame = _as_dataframe(X)
        y_array = np.asarray(y)
        self._train_X_ = frame
        self._train_y_ = y_array
        cfg = self.config
        if cfg.task_type == "classification":
            self.classes_ = np.unique(y_array)

        sample_size = _resolve_teacher_sample_size(
            cfg.teacher_sample_size,
            max_num_rows=cfg.teacher_max_num_rows,
            n_estimators=cfg.teacher_n_estimators,
        )
        self.soft_targets_ = collect_oof_targets(
            self.teacher,
            frame,
            y_array,
            task_type=cfg.task_type,
            n_folds=cfg.n_folds,
            random_state=cfg.random_state,
            classes=self.classes_,
            sample_size=sample_size,
        )

        weights: NDArray[np.float64] | None = None
        targets = self.soft_targets_
        if cfg.task_type == "classification":
            if cfg.adaptive_temperature:
                temps = adaptive_temperatures(targets, cfg.temperature)
            else:
                temps = cfg.temperature
            targets = soften_probabilities(targets, temps)
            if cfg.confidence_weighting:
                weights = confidence_weights(targets)

        self.sample_weight_ = weights
        self.student.fit(frame, y_array, soft_targets=targets, sample_weight=weights)
        teacher_idx = _context_indices(
            self._train_y_,
            np.arange(len(self._train_y_), dtype=np.intp),
            sample_size=sample_size,
            task_type=cfg.task_type,
            random_state=cfg.random_state,
        )
        self.teacher.fit(_rows(self._train_X_, teacher_idx), self._train_y_[teacher_idx])
        self.teacher_fitted_ = True
        return self

    def predict(self, X: Any) -> NDArray:
        return self.student.predict(X)

    def predict_proba(self, X: Any) -> NDArray:
        return self.student.predict_proba(X)

    def evaluate(self, X: Any, y: Any) -> dict[str, float]:
        y_array = np.asarray(y)
        if self.config.task_type == "classification":
            proba = self.predict_proba(X)
            pred = self.classes_[proba.argmax(axis=1)] if self.classes_ is not None else self.predict(X)
            return classification_metrics(y_array, pred, proba)
        return regression_metrics(y_array, self.predict(X))

    def compare(self, X: Any, y: Any) -> dict[str, Any]:
        """Student vs full-context teacher, plus a hard-label XGBoost baseline."""
        if not self.teacher_fitted_:
            raise RuntimeError("compare() requires a fitted teacher; call fit() first")
        y_array = np.asarray(y)
        result: dict[str, Any] = {"student": {}, "teacher": {}, "hard_xgb": {}}

        if self.config.task_type == "classification":
            student_pred  = self.predict(X)
            teacher_pred  = self.teacher.predict(X)
            student_proba = self.predict_proba(X)
            teacher_proba = self.teacher.predict_proba(X)
            result["student"] = classification_metrics(y_array, student_pred, student_proba)
            result["teacher"] = classification_metrics(y_array, teacher_pred, teacher_proba)
            result["retention_accuracy"] = retention(
                result["student"]["accuracy"], result["teacher"]["accuracy"]
            )
        else:
            student_pred = self.predict(X)
            teacher_pred = self.teacher.predict(X)
            result["student"] = regression_metrics(y_array, student_pred)
            result["teacher"] = regression_metrics(y_array, teacher_pred)
            result["retention_r2"] = retention(result["student"]["r2"], result["teacher"]["r2"])

        hard = self._fit_hard_baseline(X, y_array)
        result["hard_xgb"] = hard
        return result

    def _fit_hard_baseline(self, X_eval: Any, y_eval: NDArray) -> dict[str, float]:
        """Train a second XGBoost on hard labels only, using stored train data.

        The baseline is fit on the same student encoder inputs reconstructed
        from the last ``fit`` call's training table. We expose it only as a
        comparison metric, so we re-fit from the evaluation caller's training
        split by reading attributes set during ``fit``.
        """
        train_X = getattr(self, "_train_X_", None)
        train_y = getattr(self, "_train_y_", None)
        if train_X is None or train_y is None:
            return {}
        baseline = XGBoostStudent(
            task_type=self.config.task_type,
            alpha=0.0,
            random_state=self.config.random_state,
            **self.config.xgb_params,
        )
        if self.config.task_type == "classification":
            n_classes = len(self.classes_)
            dummy_soft = np.full((len(train_y), n_classes), 1.0 / n_classes)
        else:
            dummy_soft = np.asarray(train_y, dtype=np.float64)
        baseline.fit(train_X, train_y, soft_targets=dummy_soft)
        if self.config.task_type == "classification":
            proba = baseline.predict_proba(X_eval)
            pred = baseline.predict(X_eval)
            return classification_metrics(y_eval, pred, proba)
        return regression_metrics(y_eval, baseline.predict(X_eval))

    def save(self, path: str) -> None:
        payload = {
            "config": asdict(self.config),
            "student": self.student,
            "classes_": self.classes_,
            "soft_targets_": self.soft_targets_,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> TabFMDistiller:
        payload = joblib.load(path)
        config = DistillConfig(**payload["config"])
        distiller = object.__new__(cls)
        distiller.config = config
        distiller.teacher = None
        distiller.student = payload["student"]
        distiller.classes_ = payload.get("classes_")
        distiller.soft_targets_ = payload.get("soft_targets_")
        distiller.sample_weight_ = None
        distiller.teacher_fitted_ = False
        distiller._train_X_ = None
        distiller._train_y_ = None
        return distiller
