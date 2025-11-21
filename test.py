"""
test.py
-------

Script to:
- load the saved DecisionTreeClassifier model from ``saved_model.pth``
- recompute the 70/30 train/test split
- evaluate the model on the test split and print accuracy
"""

from mlops_major.model_utils import evaluate_model, load_model
from mlops_major.preprocess import get_train_test_split


def main():
    # Recreate the same split used during training
    X_train, X_test, y_train, y_test = get_train_test_split(
        test_size=0.3,
        random_state=42,
    )

    model = load_model()
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Test accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()


