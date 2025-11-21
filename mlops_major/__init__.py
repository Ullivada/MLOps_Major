"""
mlops_major
============

Utility package for the ML Ops major assignment.

This package exposes helpers to:
- load the Olivetti faces dataset
- perform train/test split
- train, evaluate and persist a DecisionTreeClassifier model
"""

from .data import get_dataset  # noqa: F401
from .preprocess import get_train_test_split  # noqa: F401
from .model_utils import (  # noqa: F401
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


