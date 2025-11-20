from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import certifi
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split

os.environ.setdefault("SSL_CERT_FILE", certifi.where())
DATA_HOME = Path(".data/sklearn")
DATA_HOME.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    target_names: Tuple[int, ...]


def load_dataset(
    test_size: float = 0.3,
    random_state: int = 42,
) -> DatasetSplit:
    """Fetch the Olivetti faces dataset and return a deterministic split."""
    dataset = fetch_olivetti_faces(data_home=str(DATA_HOME))
    x = dataset.data
    y = dataset.target

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return DatasetSplit(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        target_names=tuple(sorted(set(dataset.target))),
    )

