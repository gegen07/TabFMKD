# TabFM-KD

Knowledge distillation from [Google TabFM](https://github.com/google-research/tabfm) or [TabICL](https://github.com/soda-inria/tabicl) into a deployable **XGBoost** student.

TabFM and TabICL are zero-shot tabular foundation models: `fit()` stores labeled rows as in-context examples and `predict()` runs a single forward pass. That is accurate, but serving the foundation model is heavy (large GPU footprint, low throughput). This project transfers the teacher's probability distribution into a gradient-boosted tree so you keep most of the accuracy at CPU latency.

```
labeled table
     │
     ▼
┌─────────────────────────────┐
│  TabFM / TabICL teacher     │
│  k-fold out-of-fold soft y  │  ← avoids ICL identity leakage
└──────────────┬──────────────┘
               │  p_teacher, T, α
               ▼
┌─────────────────────────────┐
│  XGBoost student            │
│  mixed hard + soft targets  │
└─────────────────────────────┘
```

## Why out-of-fold soft labels

TabFM conditions on whatever you passed to `fit`. If you then ask it to label those same rows, the soft targets collapse toward one-hot memorization. The distiller therefore:

1. Splits the training set into `n_folds` stratified folds.
2. Fits a fresh TabFM estimator on `n_folds - 1` folds (same frozen backbone, new context).
3. Predicts probabilities on the held-out fold.
4. Trains XGBoost on the assembled out-of-fold distribution.

That is the same leakage fix used for other in-context tabular models (TabPFN, TabICL).

## Distillation recipe

Classification uses a tree-friendly form of Hinton KD:

- Temperature `T` (default 3.0) softens teacher probabilities: `p^(T) = softmax(log(p) / T)`.
- Optional **adaptive temperature** scales `T` by each row's teacher entropy.
- Optional **confidence weighting** down-weights ambiguous teacher rows.
- Mixed targets: `y_mix = α · p^(T) + (1 − α) · y_onehot` (default `α = 0.7`).

The XGBoost student then:

- **Binary:** `binary:logistic` with the mixed positive-class probability as a float label.
- **Multiclass:** a custom softmax objective against the full mixed distribution, so inter-class similarities survive.
- **Regression:** squared error on `α · ŷ_teacher + (1 − α) · y`.

`save()` strips the teacher. The joblib artifact depends only on XGBoost.

## Install

```bash
cd tabfm-kd
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Google TabFM requires **Python >= 3.11** and downloads pretrained weights from Hugging Face (non-commercial license `tabfm-non-commercial-v1.0`):

```bash
pip install -e ".[tabfm]"
```

[TabICL](https://github.com/soda-inria/tabicl) is a separate ICL backbone (Python >= 3.10). Checkpoints download from Hugging Face on first `fit`:

```bash
pip install -e ".[tabicl]"
```

On Python 3.10, or when you do not want to pull a checkpoint, use `--teacher sklearn`. That fallback is a histogram GBDT so the pipeline, CLI, and tests still run.

## Quick start

```python
from sklearn.model_selection import train_test_split
from tabfm_kd import DistillConfig, TabFMDistiller
from tabfm_kd.data import load_builtin

data = load_builtin("breast_cancer")
X_train, X_test, y_train, y_test = train_test_split(
    data.X, data.y, test_size=0.2, stratify=data.y, random_state=42
)

distiller = TabFMDistiller(
    DistillConfig(
        teacher="tabfm",          # or "tabicl" / "sklearn"
        teacher_backend="pytorch",
        teacher_n_estimators=8,   # TabFM context ensemble size
        n_folds=5,
        temperature=3.0,
        alpha=0.7,
    )
)
distiller.fit(X_train, y_train)
print(distiller.compare(X_test, y_test))
distiller.save("artifacts/student.joblib")
```

CLI:

```bash
# Local fallback — no TabFM download
tabfm-kd distill --teacher sklearn --dataset breast_cancer --output artifacts/student.joblib

# Real TabFM teacher (Python >= 3.11 + tabfm extra)
tabfm-kd distill --teacher tabfm --backend pytorch --dataset wine --output artifacts/student.joblib

# TabICL teacher
tabfm-kd distill --teacher tabicl --dataset wine --output artifacts/student.joblib

tabfm-kd evaluate --student artifacts/student.joblib --dataset breast_cancer
```

Example script:

```bash
python examples/run_distillation.py --teacher sklearn --dataset breast_cancer
python examples/run_distillation.py --teacher tabfm --dataset wine
python examples/run_distillation.py --teacher tabicl --dataset wine
```

Built-in tables: `breast_cancer`, `wine`, `iris`, `diabetes`. Any CSV works with `--csv path --target col`.

## Configuration

| Knob | Default | Role |
| --- | --- | --- |
| `teacher` | `tabfm` | `tabfm`, `tabicl`, or `sklearn` |
| `teacher_n_estimators` | 8 | Ensemble members for TabFM / TabICL (TabFM upstream default is 32; 8 is faster for KD) |
| `teacher_max_num_rows` | 100 | TabFM per-bag ICL window. Also sizes the default context pool (`max_num_rows * n_estimators`) for every teacher |
| `teacher_sample_strategy` | `proportional` | `proportional` keeps class frequencies in the ~800-row context; `balanced` equalizes them |
| `n_folds` | 5 | Out-of-fold soft-label folds |
| `temperature` | 3.0 | Hinton softening |
| `alpha` | 0.7 | Soft-label vs hard-label mix |
| `adaptive_temperature` | True | Per-row `T` from teacher entropy |
| `confidence_weighting` | True | Bell-shaped sample weights |
| `augment_factor` | None | Extra Gaussian-noised numeric rows |

`compare()` reports the teacher, the distilled student, and a hard-label XGBoost baseline plus accuracy/R² retention. Classification also includes **Gini** (`2 * ROC-AUC - 1`) and **average precision** (PR-AUC), which are more informative than accuracy on imbalanced labels.

## Layout

```
src/tabfm_kd/
  teacher.py        TabFM / TabICL wrappers + sklearn fallback
  losses.py         temperature, adaptive T, confidence weights
  student.py        XGBoost on mixed soft targets
  distillation.py   k-fold collection + orchestrator
  data.py           sklearn / CSV loaders
  metrics.py        accuracy, AUC, Gini, AP, RMSE, retention
  cli.py            tabfm-kd distill | evaluate
```

## Tests

```bash
pytest -q
```

Tests use the sklearn teacher so they do not download TabFM weights.

## License notes

This repository's code is Apache-2.0. TabFM **source** is Apache-2.0; the default pretrained weights pulled by `tabfm_v1_0_0.load()` are **not** — they are restricted to non-commercial, non-production use. Do not ship those weights into a commercial product. TabICL is a separately licensed package; see the [TabICL repository](https://github.com/soda-inria/tabicl) for its terms.
