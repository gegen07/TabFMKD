"""Teacher models: TabFM, TabICL, TabPFN, Mitra, and a local sklearn fallback.

Foundation-model ``fit`` does not train weights. It stores labeled rows as
in-context examples and runs a single forward pass at ``predict`` time. That
is why soft labels must be collected out-of-fold — see
:class:`tabfm_kd.distillation.TabFMDistiller`.
"""

from __future__ import annotations

import inspect
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


def _accepted_kwargs(cls: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the constructor does not take, unless it accepts ``**kwargs``."""
    try:
        signature = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return params
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return params
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in params.items() if key in accepted}


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


class TabICLTeacher:
    """sklearn-style wrapper around soda-inria TabICL.

    Same ICL contract as TabFM: ``fit`` stores context, ``predict`` does the
    forward pass. The pretrained ``model_`` is shared across k-fold clones so
    each fold does not reload the checkpoint. ``kv_cache`` defaults to on
    because the distiller scores the held-out fold in chunks.
    """

    def __init__(
        self,
        task_type: str = "classification",
        n_estimators: int = 8,
        batch_size: int | None = None,
        kv_cache: bool | str = True,
        random_state: int = 42,
        verbose: bool = False,
        device: str = "auto",
        **estimator_kwargs: Any,
    ) -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.device = resolve_compute_device(device)
        if batch_size is None:
            self.batch_size = 8 if str(self.device).startswith("cuda") else 1
        else:
            if batch_size < 1:
                raise ValueError("teacher_batch_size must be >= 1")
            self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.random_state = random_state
        self.verbose = verbose
        self.estimator_kwargs = estimator_kwargs
        self._backbone: Any = None
        self._model_config: Any = None
        self._model_path: Any = None
        self.estimator_: Any = None
        self.classes_: NDArray | None = None

    def _ensure_backbone(self) -> None:
        if self._backbone is not None:
            return
        probe = self._make_estimator()
        probe._load_model()
        self._backbone = probe.model_
        self._model_config = getattr(probe, "model_config_", None)
        self._model_path = getattr(probe, "model_path_", None)

    def _make_estimator(self) -> Any:
        try:
            from tabicl import TabICLClassifier, TabICLRegressor
        except ImportError as exc:
            raise ImportError(
                "TabICL is not installed. Install with `pip install 'tabfm-kd[tabicl]'`, "
                "or pass teacher='sklearn' / teacher='tabfm'."
            ) from exc
        estimator_cls = (
            TabICLClassifier if self.task_type == "classification" else TabICLRegressor
        )
        params = _accepted_kwargs(
            estimator_cls,
            _estimator_params(
                n_estimators=self.n_estimators,
                batch_size=self.batch_size,
                kv_cache=self.kv_cache,
                random_state=self.random_state,
                verbose=self.verbose,
                device=self.device,
                **self.estimator_kwargs,
            ),
        )
        estimator = estimator_cls(**params)
        if self._backbone is not None:
            backbone = self._backbone
            config = self._model_config
            path = self._model_path

            def _load_shared_model() -> None:
                estimator.model_ = backbone
                estimator.model_config_ = config
                estimator.model_path_ = path

            estimator._load_model = _load_shared_model  # type: ignore[method-assign]
        return estimator

    def fit(self, X: Any, y: Any) -> TabICLTeacher:
        self._ensure_backbone()
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)
        if self.task_type == "classification":
            self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X: Any) -> NDArray:
        if self.estimator_ is None:
            raise RuntimeError("TabICLTeacher must be fit before predict")
        return np.asarray(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> NDArray:
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")
        if self.estimator_ is None:
            raise RuntimeError("TabICLTeacher must be fit before predict_proba")
        return np.asarray(self.estimator_.predict_proba(X))

    def clone_unfitted(self) -> TabICLTeacher:
        self._ensure_backbone()
        clone = TabICLTeacher(
            task_type=self.task_type,
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            kv_cache=self.kv_cache,
            random_state=self.random_state,
            verbose=self.verbose,
            device=self.device,
            **self.estimator_kwargs,
        )
        clone._backbone = self._backbone
        clone._model_config = self._model_config
        clone._model_path = self._model_path
        return clone


_MITRA_CLS_REPO = "autogluon/mitra-classifier-2"
_MITRA_REG_REPO = "autogluon/mitra-regressor-2"


class TabPFNTeacher:
    """sklearn-style wrapper around Prior Labs TabPFN v2.

    Uses ``TabPFNClassifier.create_default_for_version(ModelVersion.V2)`` so
    this is the v2 checkpoint, not the default TabPFN-3 weights. ``fit``
    stores context; ``predict`` runs ICL. ``ignore_pretraining_limits`` is
    on so a large ``teacher_sample_size`` is not rejected.
    """

    def __init__(
        self,
        task_type: str = "classification",
        n_estimators: int = 8,
        batch_size: int | None = None,
        random_state: int = 42,
        verbose: bool = False,
        device: str = "auto",
        fit_mode: str = "fit_with_cache",
        ignore_pretraining_limits: bool = True,
        **estimator_kwargs: Any,
    ) -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.device = resolve_compute_device(device)
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose
        self.fit_mode = fit_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.estimator_kwargs = estimator_kwargs
        self.estimator_: Any = None
        self.classes_: NDArray | None = None

    def _make_estimator(self) -> Any:
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
        except ImportError as exc:
            raise ImportError(
                "TabPFN is not installed. Install with `pip install 'tabfm-kd[tabpfn]'`, "
                "or pass teacher='sklearn'."
            ) from exc
        estimator_cls = (
            TabPFNClassifier if self.task_type == "classification" else TabPFNRegressor
        )
        params = _estimator_params(
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.random_state,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            fit_mode=self.fit_mode,
            **self.estimator_kwargs,
        )
        params = _accepted_kwargs(estimator_cls, params)
        try:
            from tabpfn.constants import ModelVersion

            version = getattr(ModelVersion, "V2", None)
            factory = getattr(estimator_cls, "create_default_for_version", None)
            if version is not None and callable(factory):
                logger.info("Loading TabPFN v2 %s on %s", self.task_type, self.device)
                return factory(version, **params)
        except Exception as exc:
            logger.warning(
                "TabPFN ModelVersion.V2 factory failed (%s); using TabPFNClassifier defaults",
                exc,
            )
        logger.info("Loading TabPFN %s on %s", self.task_type, self.device)
        return estimator_cls(**params)

    def fit(self, X: Any, y: Any) -> TabPFNTeacher:
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)
        if self.task_type == "classification":
            self.classes_ = np.asarray(self.estimator_.classes_)
        return self

    def predict(self, X: Any) -> NDArray:
        if self.estimator_ is None:
            raise RuntimeError("TabPFNTeacher must be fit before predict")
        return np.asarray(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> NDArray:
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")
        if self.estimator_ is None:
            raise RuntimeError("TabPFNTeacher must be fit before predict_proba")
        return np.asarray(self.estimator_.predict_proba(X))

    def clone_unfitted(self) -> TabPFNTeacher:
        return TabPFNTeacher(
            task_type=self.task_type,
            n_estimators=self.n_estimators,
            batch_size=self.batch_size,
            random_state=self.random_state,
            verbose=self.verbose,
            device=self.device,
            fit_mode=self.fit_mode,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
            **self.estimator_kwargs,
        )


class MitraTeacher:
    """sklearn-style wrapper around AutoGluon Mitra-v2.

    Loads ``autogluon/mitra-classifier-2`` (or ``mitra-regressor-2``) with
    ``fine_tune=False`` so ``fit`` only stores context and ``predict`` runs
    ICL — the same contract as TabFM / TabICL. Pass
    ``teacher_params={"fine_tune": True}`` to run AutoGluon's fine-tune recipe.
    """

    def __init__(
        self,
        task_type: str = "classification",
        n_estimators: int = 1,
        random_state: int = 42,
        verbose: bool = False,
        device: str = "auto",
        fine_tune: bool = False,
        hf_model: str | None = None,
        **estimator_kwargs: Any,
    ) -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.device = resolve_compute_device(device)
        self.random_state = random_state
        self.verbose = verbose
        self.fine_tune = fine_tune
        self.hf_model = hf_model or (
            _MITRA_CLS_REPO if task_type == "classification" else _MITRA_REG_REPO
        )
        self.estimator_kwargs = estimator_kwargs
        self.estimator_: Any = None
        self.classes_: NDArray | None = None

    def _make_estimator(self) -> Any:
        try:
            from autogluon.tabular.models.mitra.sklearn_interface import (
                MitraClassifier,
                MitraRegressor,
            )
        except ImportError as exc:
            raise ImportError(
                "Mitra is not installed. Install with `pip install 'tabfm-kd[mitra]'` "
                "(AutoGluon tabular extra), or pass teacher='sklearn'."
            ) from exc
        estimator_cls = (
            MitraClassifier if self.task_type == "classification" else MitraRegressor
        )
        params = _accepted_kwargs(
            estimator_cls,
            _estimator_params(
                n_estimators=self.n_estimators,
                device=self.device,
                fine_tune=self.fine_tune,
                hf_model=self.hf_model,
                seed=self.random_state,
                verbose=self.verbose,
                **self.estimator_kwargs,
            ),
        )
        logger.info(
            "Loading Mitra-v2 %s from %s on %s (fine_tune=%s)",
            self.task_type,
            self.hf_model,
            self.device,
            self.fine_tune,
        )
        return estimator_cls(**params)

    def fit(self, X: Any, y: Any) -> MitraTeacher:
        self.estimator_ = self._make_estimator()
        self.estimator_.fit(X, y)
        if self.task_type == "classification":
            classes = getattr(self.estimator_, "classes_", None)
            self.classes_ = (
                np.asarray(classes) if classes is not None else np.unique(np.asarray(y))
            )
        return self

    def predict(self, X: Any) -> NDArray:
        if self.estimator_ is None:
            raise RuntimeError("MitraTeacher must be fit before predict")
        pred = np.asarray(self.estimator_.predict(X))
        if (
            self.task_type == "classification"
            and self.classes_ is not None
            and pred.dtype.kind in {"i", "u"}
            and pred.size
            and int(np.max(pred)) < len(self.classes_)
        ):
            return self.classes_[pred]
        return pred

    def predict_proba(self, X: Any) -> NDArray:
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")
        if self.estimator_ is None:
            raise RuntimeError("MitraTeacher must be fit before predict_proba")
        return np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)

    def clone_unfitted(self) -> MitraTeacher:
        return MitraTeacher(
            task_type=self.task_type,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            verbose=self.verbose,
            device=self.device,
            fine_tune=self.fine_tune,
            hf_model=self.hf_model,
            **self.estimator_kwargs,
        )


class SklearnFallbackTeacher:
    """Histogram GBDT stand-in used when foundation-model weights are unavailable."""

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


_FOUNDATION_ONLY_KEYS = {
    "backend",
    "n_estimators",
    "max_num_rows",
    "max_num_features",
    "batch_size",
    "cache_context",
    "kv_cache",
    "verbose",
    "device",
}

_TABICL_DROP_KEYS = {
    "backend",
    "max_num_rows",
    "max_num_features",
    "cache_context",
}


_TABPFN_DROP_KEYS = {
    "backend",
    "max_num_rows",
    "max_num_features",
    "kv_cache",
    "cache_context",
    "batch_size",
}

_MITRA_DROP_KEYS = {
    "backend",
    "max_num_rows",
    "max_num_features",
    "cache_context",
    "kv_cache",
    "batch_size",
}


def _tabicl_init_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    params = {key: value for key, value in kwargs.items() if key not in _TABICL_DROP_KEYS}
    if "kv_cache" not in params and "cache_context" in kwargs:
        params["kv_cache"] = kwargs["cache_context"]
    return params


def _tabpfn_init_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    params = {key: value for key, value in kwargs.items() if key not in _TABPFN_DROP_KEYS}
    if "fit_mode" not in params and kwargs.get("cache_context") is False:
        params["fit_mode"] = "fit_preprocessors"
    return params


def _mitra_init_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in kwargs.items() if key not in _MITRA_DROP_KEYS}


def build_teacher(
    name: str,
    *,
    task_type: str = "classification",
    **kwargs: Any,
) -> Teacher:
    key = name.lower().strip().replace("-", "").replace("_", "")
    if key in {"sklearn", "fallback"}:
        filtered = {
            k: v for k, v in kwargs.items() if k not in _FOUNDATION_ONLY_KEYS
        }
        return SklearnFallbackTeacher(task_type=task_type, **filtered)
    if key == "tabicl":
        return TabICLTeacher(task_type=task_type, **_tabicl_init_kwargs(kwargs))
    if key in {"tabpfn", "tabpfnv2", "pfnv2", "pfn"}:
        return TabPFNTeacher(task_type=task_type, **_tabpfn_init_kwargs(kwargs))
    if key in {"mitra", "mitrav2"}:
        return MitraTeacher(task_type=task_type, **_mitra_init_kwargs(kwargs))
    if key == "tabfm":
        return TabFMTeacher(task_type=task_type, **kwargs)
    raise ValueError(f"Unknown teacher '{name}'")
