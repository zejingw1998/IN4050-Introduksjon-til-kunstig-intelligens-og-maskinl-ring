import numpy as np


class FlattenTransformer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X.reshape(X.shape[0], -1)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class PixelNormalizer:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32) / 255.0

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def apply_pipeline(X, transformers):
    result = X
    for transformer in transformers:
        result = transformer.transform(result)
    return result


def stratified_subset_indices(y, size, rng):
    y = np.asarray(y)
    classes = np.unique(y)
    per_class = size // len(classes)
    remainder = size % len(classes)
    chosen = []

    for i, cls in enumerate(classes):
        cls_indices = np.flatnonzero(y == cls)
        count = per_class + (1 if i < remainder else 0)
        count = min(count, len(cls_indices))
        chosen.extend(rng.choice(cls_indices, size=count, replace=False).tolist())

    rng.shuffle(chosen)
    return np.array(chosen, dtype=int)
