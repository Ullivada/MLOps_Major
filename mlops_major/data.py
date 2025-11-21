from sklearn.datasets import fetch_olivetti_faces


def get_dataset():
    """
    Load the Olivetti faces dataset.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Flattened face images.
    y : ndarray of shape (n_samples,)
        Target labels (person IDs).
    """
    dataset = fetch_olivetti_faces()
    return dataset.data, dataset.target


__all__ = ["get_dataset"]


