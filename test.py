from __future__ import annotations

import json

from mlops_major.data import load_dataset
from mlops_major.model_utils import evaluate_saved_model


def main() -> None:
    dataset = load_dataset()
    results = evaluate_saved_model(dataset)
    print("Evaluation results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

