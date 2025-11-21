from sklearn.model_selection import train_test_split

from .data import get_dataset


def get_train_test_split(test_size: float = 0.3, random_state: int = 42):
    """
    Prepare a train/test split of the Olivetti faces dataset.

    Parameters
    ----------
    test_size : float, default=0.3
        Proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random seed to make the split reproducible across scripts.

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Train/test split of features and targets.
    """
    X, y = get_dataset()
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


__all__ = ["get_train_test_split"]


