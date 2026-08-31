import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from tabfm_kd.student import XGBoostStudent


def test_binary_student_learns_soft_labels():
    X, y = make_classification(
        n_samples=160,
        n_features=8,
        n_informative=6,
        random_state=0,
    )
    soft = np.zeros((len(y), 2))
    soft[np.arange(len(y)), y] = 0.85
    soft[np.arange(len(y)), 1 - y] = 0.15
    student = XGBoostStudent(
        task_type="classification",
        alpha=0.8,
        n_estimators=40,
        max_depth=3,
        random_state=0,
    )
    student.fit(X, y, soft_targets=soft)
    proba = student.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (student.predict(X) == y).mean() > 0.85


def test_multiclass_student_custom_objective():
    X, y = make_classification(
        n_samples=180,
        n_features=10,
        n_informative=7,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=1,
    )
    n_classes = 3
    soft = np.full((len(y), n_classes), 0.05)
    soft[np.arange(len(y)), y] = 0.9
    student = XGBoostStudent(
        task_type="classification",
        alpha=0.7,
        n_estimators=50,
        max_depth=3,
        random_state=1,
    )
    student.fit(X, y, soft_targets=soft)
    proba = student.predict_proba(X)
    assert proba.shape == (len(y), 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (student.predict(X) == y).mean() > 0.8


def test_regression_student_and_mixed_types():
    X_num, y = make_regression(n_samples=80, n_features=4, random_state=2)
    frame = pd.DataFrame(X_num, columns=list("abcd"))
    frame["cat"] = np.where(frame["a"] > 0, "pos", "neg")
    student = XGBoostStudent(
        task_type="regression",
        alpha=0.5,
        n_estimators=30,
        max_depth=3,
        random_state=2,
    )
    student.fit(frame, y, soft_targets=y)
    pred = student.predict(frame)
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()
