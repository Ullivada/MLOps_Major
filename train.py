from __future__ import annotations

import argparse
import json

from mlops_major.data import load_dataset
from mlops_major.model_utils import TrainingConfig, train_and_save


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DecisionTree on Olivetti faces.")
    parser.add_argument("--max-depth", type=int, default=None, help="Max depth for tree")
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="Minimum samples per leaf node",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducible results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset()
    config = TrainingConfig(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
    )
    metadata = train_and_save(dataset, config=config)
    print("Training completed. Metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

