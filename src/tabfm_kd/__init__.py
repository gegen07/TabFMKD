"""Knowledge distillation from Google TabFM into an XGBoost student."""

from tabfm_kd.distillation import DistillConfig, TabFMDistiller
from tabfm_kd.student import XGBoostStudent
from tabfm_kd.teacher import SklearnFallbackTeacher, TabFMTeacher, build_teacher

__version__ = "0.1.0"

__all__ = [
    "DistillConfig",
    "TabFMDistiller",
    "XGBoostStudent",
    "TabFMTeacher",
    "SklearnFallbackTeacher",
    "build_teacher",
]
