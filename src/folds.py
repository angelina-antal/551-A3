import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_stratified_folds(
    y: np.ndarray,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
):
    y = np.asarray(y)
    dummy_x = np.zeros(len(y))  # skf needs X, but we only stratify on y

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    return list(skf.split(dummy_x, y))