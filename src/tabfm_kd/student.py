"""XGBoost student trained on mixed hard/soft teacher targets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from tabfm_kd.losses import mix_hard_soft, mix_regression_targets, one_hot, softmax_from_logits


def _as_dataframe(X: Any) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    array = np.asarray(X)
    columns = [f"f{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


class FeatureEncoder:
    """Median-impute numerics and ordinal-encode categoricals for XGBoost."""

    def __init__(self) -> None:
        self._pipeline: ColumnTransformer | Pipeline | None = None

    def fit_transform(self, X: Any) -> NDArray[np.float32]:
        frame = _as_dataframe(X)
        numeric_cols = list(frame.select_dtypes(include=[np.number]).columns)
        categorical_cols = [col for col in frame.columns if col not in numeric_cols]

        transformers: list[tuple[str, Any, list[str]]] = []
        if numeric_cols:
            transformers.append(
                ("num", SimpleImputer(strategy="median"), numeric_cols)
            )
        if categorical_cols:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "encode",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    categorical_cols,
                )
            )
        if not transformers:
            raise ValueError("No features available to encode")
        self._pipeline = ColumnTransformer(transformers, remainder="drop")
        return np.asarray(self._pipeline.fit_transform(frame), dtype=np.float32)

    def transform(self, X: Any) -> NDArray[np.float32]:
        if self._pipeline is None:
            raise RuntimeError("FeatureEncoder must be fit first")
        frame = _as_dataframe(X)
        return np.asarray(self._pipeline.transform(frame), dtype=np.float32)


def _multiclass_soft_objective(targets: NDArray[np.float64]):
    """Softmax cross-entropy against mixed teacher/student targets.

    XGBoost packs multiclass margins as class-major ``(n_classes, n_samples)``.
    The diagonal Hessian ``p (1 - p)`` is the usual tree approximation.
    """
    n_samples, n_classes = targets.shape

    def objective(preds: NDArray[np.floating], _dtrain: xgb.DMatrix):
        logits = np.reshape(preds, (n_classes, n_samples), order="C").T
        probs = softmax_from_logits(logits)
        grad = (probs - targets).T.ravel()
        hess = np.maximum((probs * (1.0 - probs)).T.ravel(), 1e-6)
        return grad, hess

    return objective


class XGBoostStudent:
    """Tree student that consumes TabFM soft targets.

    * Binary classification uses ``binary:logistic`` with float labels in
      ``[0, 1]`` — the mixed positive-class probability.
    * Multiclass uses a custom softmax objective against the full mixed
      distribution so inter-class similarities survive distillation.
    * Regression fits squared error on ``alpha * y_teacher + (1-alpha) * y``.
    """

    def __init__(
        self,
        task_type: str = "classification",
        alpha: float = 0.7,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        n_estimators: int = 200,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 1.0,
        reg_lambda: float = 1.0,
        tree_method: str = "hist",
        random_state: int = 42,
        verbose: bool = False,
        **xgb_params: Any,
    ) -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be 'classification' or 'regression'")
        self.task_type = task_type
        self.alpha = alpha
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.tree_method = tree_method
        self.random_state = random_state
        self.verbose = verbose
        self.xgb_params = xgb_params
        self.encoder_ = FeatureEncoder()
        self.booster_: xgb.Booster | None = None
        self.classes_: NDArray | None = None
        self.n_classes_: int | None = None
        self._uses_custom_obj = False

    def _base_params(self) -> dict[str, Any]:
        params = {
            "max_depth": self.max_depth,
            "eta": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "seed": self.random_state,
            "verbosity": 1 if self.verbose else 0,
        }
        params.update(self.xgb_params)
        return params

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        soft_targets: NDArray[np.floating],
        sample_weight: NDArray[np.floating] | None = None,
    ) -> XGBoostStudent:
        features = self.encoder_.fit_transform(X)
        y_array = np.asarray(y)
        params = self._base_params()
        self._uses_custom_obj = False

        if self.task_type == "regression":
            labels = mix_regression_targets(soft_targets, y_array, self.alpha)
            dtrain = xgb.DMatrix(features, label=labels, weight=sample_weight)
            params["objective"] = "reg:squarederror"
            self.booster_ = xgb.train(
                params, dtrain, num_boost_round=self.n_estimators
            )
            return self

        classes, encoded = np.unique(y_array, return_inverse=True)
        self.classes_ = classes
        self.n_classes_ = int(len(classes))
        mixed = mix_hard_soft(
            np.asarray(soft_targets, dtype=np.float64),
            one_hot(encoded, self.n_classes_),
            self.alpha,
        )

        if self.n_classes_ == 2:
            dtrain = xgb.DMatrix(
                features, label=mixed[:, 1], weight=sample_weight
            )
            params["objective"] = "binary:logistic"
            params["eval_metric"] = "logloss"
            self.booster_ = xgb.train(
                params, dtrain, num_boost_round=self.n_estimators
            )
            return self

        dtrain = xgb.DMatrix(features, label=encoded, weight=sample_weight)
        params["num_class"] = self.n_classes_
        params["disable_default_eval_metric"] = 1
        self._uses_custom_obj = True
        self.booster_ = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            obj=_multiclass_soft_objective(mixed),
        )
        return self

    def _raw_predict(self, X: Any) -> NDArray[np.float64]:
        if self.booster_ is None:
            raise RuntimeError("XGBoostStudent must be fit before predict")
        dtest = xgb.DMatrix(self.encoder_.transform(X))
        return np.asarray(self.booster_.predict(dtest), dtype=np.float64)

    def predict_proba(self, X: Any) -> NDArray[np.float64]:
        if self.task_type != "classification":
            raise AttributeError("predict_proba is only available for classification")
        raw = self._raw_predict(X)
        if self.n_classes_ == 2:
            if raw.ndim == 2:
                return raw
            positive = raw.reshape(-1)
            return np.column_stack([1.0 - positive, positive])
        if raw.ndim == 2:
            return softmax_from_logits(raw) if self._uses_custom_obj else raw
        n_samples = raw.size // int(self.n_classes_)
        logits = np.reshape(raw, (int(self.n_classes_), n_samples), order="C").T
        return softmax_from_logits(logits)

    def predict(self, X: Any) -> NDArray:
        if self.task_type == "regression":
            return self._raw_predict(X).reshape(-1)
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
