import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import StratifiedKFold, train_test_split

from tabfm_kd.distillation import (
    DistillConfig,
    TabFMDistiller,
    _fmt_duration,
    _resolve_teacher_sample_size,
    _subsample_indices,
    collect_oof_targets,
)
from tabfm_kd.teacher import (
    MitraTeacher,
    SklearnFallbackTeacher,
    TabICLTeacher,
    TabPFNTeacher,
    build_teacher,
    resolve_compute_device,
    resolve_teacher_batch_size,
)


class _RecordingTeacher:
    """Stub that records which rows were passed to ``fit``."""

    def __init__(self, calls: list | None = None) -> None:
        self.task_type = "classification"
        self.classes_: np.ndarray | None = None
        self.fit_row_ids = calls if calls is not None else []

    def clone_unfitted(self) -> "_RecordingTeacher":
        clone = _RecordingTeacher(self.fit_row_ids)
        return clone

    def fit(self, X, y):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.fit_row_ids.append(frame["row_id"].to_numpy())
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X):
        n = len(X)
        k = len(self.classes_) if self.classes_ is not None else 2
        return np.full((n, k), 1.0 / k)


def test_oof_targets_never_use_full_training_set_as_context():
    """Each row is labeled by a teacher fit on the complementary folds."""
    X, y = make_classification(
        n_samples=60,
        n_features=6,
        n_informative=4,
        random_state=3,
    )
    teacher = SklearnFallbackTeacher(random_state=3, max_iter=20)
    frame = pd.DataFrame(X)
    soft = collect_oof_targets(
        teacher,
        frame,
        y,
        task_type="classification",
        n_folds=3,
        random_state=3,
    )
    assert soft.shape == (60, 2)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-6)
    # OOF labels should not be a perfect one-hot copy of y.
    hard = np.zeros_like(soft)
    hard[np.arange(len(y)), y] = 1.0
    assert not np.allclose(soft, hard)


def test_oof_teacher_is_fit_on_a_sample_of_the_train_fold():
    """Huge folds must not be copied into teacher context — only a sample."""
    n_samples = 80
    sample_size = 12
    n_folds = 4
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        random_state=7,
    )
    frame = pd.DataFrame(X)
    frame.insert(0, "row_id", np.arange(n_samples))
    teacher = _RecordingTeacher()
    soft = collect_oof_targets(
        teacher,
        frame,
        y,
        task_type="classification",
        n_folds=n_folds,
        random_state=7,
        sample_size=sample_size,
    )
    assert soft.shape == (n_samples, 2)
    assert len(teacher.fit_row_ids) == n_folds

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=7)
    for (train_idx, valid_idx), fitted in zip(splitter.split(frame, y), teacher.fit_row_ids):
        assert len(fitted) == sample_size
        assert set(fitted).issubset(set(train_idx))
        assert set(fitted).isdisjoint(set(valid_idx))
        # Stratified sample keeps both classes when there is room.
        fitted_y = y[fitted]
        assert len(np.unique(fitted_y)) == 2


def test_teacher_sample_size_none_matches_tabfm_context_budget():
    assert _resolve_teacher_sample_size(None, max_num_rows=100, n_estimators=8) == 800
    assert _resolve_teacher_sample_size(0, max_num_rows=100, n_estimators=8) is None
    assert _resolve_teacher_sample_size(50, max_num_rows=100, n_estimators=8) == 50


def test_balanced_context_equalizes_classes_when_both_are_large_enough():
    y = np.array([0] * 90 + [1] * 10)
    idx = np.arange(len(y), dtype=np.intp)
    rng = np.random.default_rng(0)
    chosen = _subsample_indices(
        y, idx, 20, "classification", rng, strategy="balanced"
    )
    assert len(chosen) == 20
    counts = np.bincount(y[chosen])
    assert counts[0] == 10
    assert counts[1] == 10


def test_balanced_context_uses_all_minority_rows_when_scarce():
    y = np.array([0] * 95 + [1] * 5)
    idx = np.arange(len(y), dtype=np.intp)
    rng = np.random.default_rng(1)
    chosen = _subsample_indices(
        y, idx, 20, "classification", rng, strategy="balanced"
    )
    assert len(chosen) == 20
    counts = np.bincount(y[chosen])
    assert counts[1] == 5
    assert counts[0] == 15


def test_oof_balanced_context_is_class_equal_and_stays_in_train_fold():
    n_samples = 100
    sample_size = 20
    n_folds = 4
    rng = np.random.default_rng(11)
    y = np.array([0] * 90 + [1] * 10)
    X = pd.DataFrame(
        {
            "row_id": np.arange(n_samples),
            "f0": rng.normal(size=n_samples),
        }
    )
    teacher = _RecordingTeacher()
    collect_oof_targets(
        teacher,
        X,
        y,
        task_type="classification",
        n_folds=n_folds,
        random_state=11,
        sample_size=sample_size,
        sample_strategy="balanced",
    )
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=11)
    for (train_idx, valid_idx), fitted in zip(splitter.split(X, y), teacher.fit_row_ids):
        assert len(fitted) == sample_size
        assert set(fitted).issubset(set(train_idx))
        assert set(fitted).isdisjoint(set(valid_idx))
        fitted_y = y[fitted]
        n_pos_train = int((y[train_idx] == 1).sum())
        n_pos_fit = int((fitted_y == 1).sum())
        assert n_pos_fit == min(n_pos_train, sample_size // 2)
        assert int((fitted_y == 0).sum()) == sample_size - n_pos_fit


def test_end_to_end_classification_distillation(tmp_path):
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        random_state=4,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4, stratify=y
    )
    distiller = TabFMDistiller(
        DistillConfig(
            task_type="classification",
            teacher="sklearn",
            n_folds=3,
            temperature=2.0,
            alpha=0.6,
            xgb_params={"n_estimators": 40, "max_depth": 3},
            random_state=4,
            device="cpu",
        )
    )
    distiller.fit(X_train, y_train)
    metrics = distiller.evaluate(X_test, y_test)
    assert metrics["accuracy"] > 0.7
    comparison = distiller.compare(X_test, y_test)
    assert "teacher" in comparison and "student" in comparison
    assert comparison["hard_xgb"]
    cm = comparison["student"]["confusion_matrix"]
    assert cm["labels"]
    assert len(cm["matrix"]) == len(cm["labels"])

    path = tmp_path / "student.joblib"
    distiller.save(str(path))
    loaded = TabFMDistiller.load(str(path))
    loaded_metrics = loaded.evaluate(X_test, y_test)
    assert loaded_metrics["accuracy"] == metrics["accuracy"]


def test_end_to_end_regression_distillation():
    X, y = make_regression(n_samples=90, n_features=5, random_state=5)
    distiller = TabFMDistiller(
        DistillConfig(
            task_type="regression",
            teacher="sklearn",
            n_folds=3,
            alpha=0.5,
            xgb_params={"n_estimators": 30, "max_depth": 3},
            random_state=5,
            device="cpu",
        )
    )
    distiller.fit(X, y)
    pred = distiller.predict(X)
    assert pred.shape == (len(y),)
    assert np.isfinite(pred).all()


def test_oof_chunked_prediction_covers_every_row():
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=8,
    )
    frame = pd.DataFrame(X)
    teacher = SklearnFallbackTeacher(random_state=8, max_iter=15)
    soft = collect_oof_targets(
        teacher,
        frame,
        y,
        task_type="classification",
        n_folds=4,
        random_state=8,
        chunk_size=7,
    )
    assert soft.shape == (40, 2)
    assert np.allclose(soft.sum(axis=1), 1.0, atol=1e-6)


def test_resolve_compute_device_and_batch_size():
    assert resolve_compute_device("cpu") == "cpu"
    assert resolve_teacher_batch_size(None, "cpu") == 1
    assert resolve_teacher_batch_size(None, "cuda") == 32
    assert resolve_teacher_batch_size(8, "cuda") == 8
    assert _fmt_duration(0) == "0s"
    assert _fmt_duration(75) == "1m 15s"
    assert _fmt_duration(3661) == "1h 01m 01s"


def test_build_teacher_routes_tabicl_and_sklearn():
    tabicl = build_teacher("tabicl", task_type="classification", device="cpu")
    assert isinstance(tabicl, TabICLTeacher)
    assert tabicl.batch_size == 1
    sklearn_teacher = build_teacher(
        "sklearn",
        task_type="classification",
        backend="pytorch",
        n_estimators=8,
        max_num_rows=100,
        batch_size=32,
        device="cpu",
    )
    assert isinstance(sklearn_teacher, SklearnFallbackTeacher)
    pfn = build_teacher("tabpfn", task_type="classification", device="cpu")
    assert isinstance(pfn, TabPFNTeacher)
    mitra = build_teacher("mitra-v2", task_type="classification", device="cpu")
    assert isinstance(mitra, MitraTeacher)
    assert mitra.hf_model == "autogluon/mitra-classifier-2"
    with pytest.raises(ValueError, match="Unknown teacher"):
        build_teacher("nope")


def test_tabicl_teacher_defers_package_import():
    teacher = TabICLTeacher(device="cpu")
    distiller = TabFMDistiller(DistillConfig(teacher="tabicl", device="cpu"))
    assert isinstance(distiller.teacher, TabICLTeacher)
    try:
        import tabicl  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="tabfm-kd\\[tabicl\\]"):
            teacher._make_estimator()
    else:
        assert teacher._make_estimator() is not None


def test_tabpfn_and_mitra_teachers_defer_package_import():
    pfn = TabPFNTeacher(device="cpu")
    mitra = MitraTeacher(device="cpu")
    assert isinstance(
        TabFMDistiller(DistillConfig(teacher="tabpfn", device="cpu")).teacher,
        TabPFNTeacher,
    )
    assert isinstance(
        TabFMDistiller(DistillConfig(teacher="mitra", device="cpu")).teacher,
        MitraTeacher,
    )
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="tabfm-kd\\[tabpfn\\]"):
            pfn._make_estimator()
    try:
        import autogluon.tabular  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="tabfm-kd\\[mitra\\]"):
            mitra._make_estimator()
