import heapq
import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 3, metric: str = "euclidean", weight: str = "uniform"):
        self.k = k
        self.metric = metric
        self.weight = weight
        self.X = None
        self.y = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X = X.copy()
        self.y = y.copy()
        if self.k <= 0 or self.k > len(self.X):
            self.k = len(self.X)

    def predict(self, X: pd.DataFrame):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame):
        predictions = []

        for _, test in X.iterrows():
            distances = np.array([self._get_metric(test.values, train.values)
                                  for _, train in self.X.iterrows()])

            nearest_idx = distances.argsort()[:self.k]
            nearest_labels = self.y.iloc[nearest_idx].values
            nearest_dist = distances[nearest_idx]

            pred = None
            if self.weight == "uniform":
                pred = nearest_labels.mean()

            elif self.weight == "rank":
                ranks = 1 / np.arange(1, self.k + 1)
                pred = np.sum(nearest_labels * ranks) / ranks.sum()

            elif self.weight == "distance":
                if np.any(nearest_dist == 0):
                    pred = nearest_labels[nearest_dist == 0][0]
                else:
                    w = 1 / nearest_dist
                    pred = np.sum(nearest_labels * w) / w.sum()

            predictions.append(pred)

        return np.array(predictions)

    def _get_metric(self, X1: np.array, X2: np.array):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((X1 - X2) ** 2))
        if self.metric == "manhattan":
            return np.sum(np.abs(X1 - X2))
        if self.metric == "chebyshev":
            return np.max(np.abs(X1 - X2))
        if self.metric == "cosine":
            return 1 - np.dot(X1, X2) / (np.linalg.norm(X1) * np.linalg.norm(X2))

    def _get_cosine(self, X1: np.array, X2: np.array):
        # Косинусное расстояние
        return 1 - np.sum(X1 * X2) / (np.sqrt(np.sum(X1 ** 2)) * np.sqrt(np.sum(X2 ** 2)))
