"""Teacher models: Google TabFM plus a local sklearn fallback.

TabFM ``fit`` does not train weights. It stores labeled rows as in-context
examples and runs a single forward pass at ``predict`` time. That is why
soft labels must be collected out-of-fold — see :class:`tabfm_kd.distillation.TabFMDistiller`.

The pretrained TabFM weights are released under ``tabfm-non-commercial-v1.0``
and require Python >= 3.11. :class:`SklearnFallbackTeacher` exists so the
distillation pipeline can be exercised without downloading those weights.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def cuda_is_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def resolve_compute_device(device: str | None = "auto") -> str:
    """Map ``auto`` / ``gpu`` / ``cuda`` / ``cpu`` to a concrete device string."""
    requested = (device or "auto").strip().lower()
    if requested in {"auto", ""}:
        return "cuda" if cuda_is_available() else "cpu"
    if requested == "gpu":
        requested = "cuda"
    if requested == "cpu":
        return "cpu"
    if requested == "cuda" or requested.startswith("cuda:"):
        if not cuda_is_available():
            logger.warning("CUDA was requested but is not available; using CPU")
            return "cpu"
        return requested
    return requested


def resolve_teacher_batch_size(batch_size: int | None, device: str) -> int:
    if batch_size is not None:
        if batch_size < 1:
            raise ValueError("teacher_batch_size must be >= 1")
        return batch_size
    return 32 if str(device).startswith("cuda") else 1


class Teacher(Protocol):
    """Minimal interface required by :class:`TabFMDistiller`."""

    task_type: str
    classes_: NDArray | None

    def fit(self, X: Any, y: Any) -> Teacher: ...

    def predict(self, X: Any) -> NDArray: ...

    def predict_proba(self, X: Any) -> NDArray: ...

    def clone_unfitted(self) -> Teacher: ...


def _estimator_params(**params: Any) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


class TabFMTeacher:
    """sklearn-style wrapper around a frozen Google TabFM backbone.

    The pretrained checkpoint is loaded once and reused across k-fold
    clones. Each clone is a fresh ``TabFMClassifier`` / ``TabFMRegressor``
    so context rows from one fold never leak into another.
    """

    def __init__(
        self,
        task_type: str = "classification",
        backend: str = "pytorch",
        n_estimators: int = 8,
        max_num_rows: int | None = 100,
        max_num_features: int | None = 500,
        batch_size: int | None = None,
        cache_context: bool = True,
        random_state: int = 42,
        verbose: bool = False,
        device: str = "auto",
        **estimator_kwargs: Any,
    ) -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        if backend not in {"pytorch", "jax"}:
            raise ValueError("backend must be 'pytorch' or 'jax'")
        self.task_type = task_type
        self.backend = backend
        self.n_estimators = n_estimators
        self.max_num_rows = max_num_rows
        self.max_num_features = max_num_features
        self.device = resolve_compute_device(device)
        self.batch_size = resolve_teacher_batch_size(batch_size, self.device)
        self.cache_context = cache_context
        self.random_state = random_state
        self.verbose = verbose
        self.estimator_kwargs = estimator_kwargs
        self._backbone: Any = None
        self.estimator_: Any = None
        self.classes_: NDArray | None = None

    def _place_backbone(self, backbone: Any) -> Any:
        if self.device == "cpu":
            return backbone
        if hasattr(backbone, "to"):
            try:
                return backbone.to(self.device)
            except (TypeError, RuntimeError, ValueError):
                logger.warning("Could not move TabFM backbone to %s via .to()", self.device)
        if str(self.device).startswith("cuda") and hasattr(backbone, "cuda"):
            try:
                return backbone.cuda()
            except (TypeError, RuntimeError, ValueError):
                logger.warning("Could not move TabFM backbone to CUDA via .cuda()")
        return backbone

    def _load_backbone(self) -> Any:
        if self._backbone is not None:
            return self._backbone
        try:
            if self.backend == "pytorch":
                from tabfm import tabfm_v1_0_0_pytorch as tabfm_loader
            else:
                from tabfm import tabfm_v1_0_0_jax as tabfm_loader
        except ImportError as exc:
            raise ImportError(
                "Google TabFM is not installed or the selected backend is missing. "
                "Install with `pip install 'tabfm-kd[tabfm]'` on Python >= 3.11, "
                "or pass teacher='sklearn' to use the local fallback."
            ) from exc
        model_type = (
            "classification" if self.task_type == "classification" else "regression"
        )
        logger.info(
            "Loading TabFM %s backbone (%s) on %s",
            model_type,
            self.backend,
            self.device,
        )
        load_kwargs: dict[str, Any] = {"model_type": model_type}
        if self.backend == "pytorch":
            load_kwargs["device"] = self.device
        try:
            backbone = tabfm_loader.load(**load_kwargs)
        except TypeError:
            backbone = tabfm_loader.load(model_type=model_type)
        self._backbone = self._place_backbone(backbone)
        return self._backbone

    def _make_estimator(self) -> Any:
        from tabfm import TabFMClassifier, TabFMRegressor

        params = _estimator_params(
            n_estimators=self.n_estimators,
            max_num_rows=self.max_num_rows,
            max_num_features=self.max_num_features,
            batch_size=self.batch_size,
            random_state=self.random_state,
            verbose=self.verbose,
            **self.estimator_kwargs,
        )
        # Context caching is a PyTorch-only speedup.
        if self.backend == "pytorch":
            params["cache_context"] = self.cache_context
        estimator_cls = (
            TabFMClassifier if self.task_type == "classification" else TabFMRegressor
        )
        return estimator_cls(model=self._load_backbone(), **params)

    def fit(self, X: Any, y: Any) -> TabFMTeacher:
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)
        if self.task_type == "classification":
            self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X: Any) -> NDArray:
        if self.estimator_ is None:
            raise RuntimeError("TabFMTeacher must be fit before predict")
        return np.asarray(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> NDArray:
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")
        if self.estimator_ is None:
            raise RuntimeError("TabFMTeacher must be fit before predict_proba")
        return np.asarray(self.estimator_.predict_proba(X))

    def clone_unfitted(self) -> TabFMTeacher:
        clone = TabFMTeacher(
            task_type=self.task_type,
            backend=self.backend,
            n_estimators=self.n_estimators,
            max_num_rows=self.max_num_rows,
            max_num_features=self.max_num_features,
            batch_size=self.batch_size,
            cache_context=self.cache_context,
            random_state=self.random_state,
            verbose=self.verbose,
            device=self.device,
            **self.estimator_kwargs,
        )
        clone._backbone = self._backbone
        return clone


_TABFM_ONLY_KEYS = {
    "backend",
    "n_estimators",
    "max_num_rows",
    "max_num_features",
    "batch_size",
    "cache_context",
    "verbose",
    "device",
}


class SklearnFallbackTeacher:
    """Histogram GBDT stand-in used when TabFM weights are unavailable."""

    def __init__(
        self,
        task_type: str = "classification",
        random_state: int = 42,
        max_iter: int = 100,
        **estimator_kwargs: Any,
    ) -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.random_state = random_state
        self.max_iter = max_iter
        self.estimator_kwargs = estimator_kwargs
        self.estimator_: Any = None
        self.classes_: NDArray | None = None

    def _make_estimator(self) -> Any:
        from sklearn.ensemble import (
            HistGradientBoostingClassifier,
            HistGradientBoostingRegressor,
        )

        params = {
            "random_state": self.random_state,
            "max_iter": self.max_iter,
            **self.estimator_kwargs,
        }
        if self.task_type == "classification":
            return HistGradientBoostingClassifier(**params)
        return HistGradientBoostingRegressor(**params)

    def fit(self, X: Any, y: Any) -> SklearnFallbackTeacher:
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)
        if self.task_type == "classification":
            self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X: Any) -> NDArray:
        if self.estimator_ is None:
            raise RuntimeError("SklearnFallbackTeacher must be fit before predict")
        return np.asarray(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> NDArray:
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")
        if self.estimator_ is None:
            raise RuntimeError("SklearnFallbackTeacher must be fit before predict_proba")
        return np.asarray(self.estimator_.predict_proba(X))

    def clone_unfitted(self) -> SklearnFallbackTeacher:
        return SklearnFallbackTeacher(
            task_type=self.task_type,
            random_state=self.random_state,
            max_iter=self.max_iter,
            **self.estimator_kwargs,
        )


def build_teacher(
    name: str,
    *,
    task_type: str = "classification",
    **kwargs: Any,
) -> Teacher:
    key = name.lower().strip()
    if key in {"sklearn", "fallback"}:
        filtered = {k: v for k, v in kwargs.items() if k not in _TABFM_ONLY_KEYS}
        return SklearnFallbackTeacher(task_type=task_type, **filtered)
    if key == "tabfm":
        return TabFMTeacher(task_type=task_type, **kwargs)
    raise ValueError(f"Unknown teacher '{name}'")
