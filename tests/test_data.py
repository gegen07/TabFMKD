import pandas as pd

from tabfm_kd.data import load_table, train_eval_split


def test_load_table_uses_passed_task_type(tmp_path):
    path = tmp_path / "churn.csv"
    frame = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "spend": [10.0, 2.0, 8.0, 3.0, 9.0, 1.0],
            "plan": ["pro", "free", "pro", "free", "pro", "free"],
            "churn": ["yes", "no", "no", "yes", "no", "yes"],
        }
    )
    frame.to_csv(path, index=False)
    bundle = load_table(str(path), target="churn", task_type="classification", drop=["id"])
    assert bundle.task_type == "classification"
    assert list(bundle.X.columns) == ["spend", "plan"]
    assert "id" not in bundle.X.columns
    X_train, X_test, y_train, y_test = train_eval_split(bundle, test_size=0.33, seed=0)
    assert len(y_train) + len(y_test) == len(bundle.y)


def test_load_table_regression_from_cmd(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame(
        {"sqft": [1.0, 2.0, 3.0, 4.0], "price": [100.5, 210.2, 305.0, 412.8]}
    ).to_csv(path, index=False)
    bundle = load_table(str(path), target="price", task_type="regression")
    assert bundle.task_type == "regression"
