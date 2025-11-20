from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import joblib
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from .data import DatasetSplit, load_dataset

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "savedmodel.pth"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"


@dataclass
class TrainingConfig:
    max_depth: int | None = None
    min_samples_leaf: int = 1
    random_state: int = 42


def train_and_save(
    dataset: DatasetSplit | None = None,
    *,
    config: TrainingConfig | None = None,
) -> Dict[str, Any]:
    """Train the DecisionTreeClassifier and persist artifacts."""
    dataset = dataset or load_dataset()
    config = config or TrainingConfig()

    model = DecisionTreeClassifier(
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        random_state=config.random_state,
    )
    model.fit(dataset.x_train, dataset.y_train)

    predictions = model.predict(dataset.x_test)
    accuracy = accuracy_score(dataset.y_test, predictions)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "accuracy": accuracy,
        "hyperparameters": asdict(config),
        "n_train": int(len(dataset.x_train)),
        "n_test": int(len(dataset.x_test)),
        "target_names": [int(x) for x in dataset.target_names],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))

    return metadata


def evaluate_saved_model(dataset: DatasetSplit | None = None) -> Dict[str, Any]:
    """Load the saved artifact and compute accuracy on the test set."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train.py before evaluating."
        )

    dataset = dataset or load_dataset()

    model: DecisionTreeClassifier = joblib.load(MODEL_PATH)
    predictions = model.predict(dataset.x_test)
    accuracy = accuracy_score(dataset.y_test, predictions)

    return {
        "accuracy": accuracy,
        "n_test": len(dataset.x_test),
    }

