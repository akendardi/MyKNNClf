import heapq
import numpy as np
import pandas as pd


class MyKNNClf:
    def __init__(self, k: int = 0, metric: str = "euclidean", weight: str = "uniform"):
        # Количество ближайших соседей
        self.k = k
        # Выбранная метрика расстояния
        self.metric = metric
        # Схема взвешивания соседей
        self.weight = weight
        # Размер обучающей выборки
        self.train_size = None
        # Матрица признаков обучающей выборки
        self.X = None
        # Метки классов обучающей выборки
        self.y = None

    def __str__(self):
        # Строковое представление объекта
        return f"MyKNNClf class: k={self.k}"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Сохраняем размер выборки
        self.train_size = (X.shape[0], X.shape[1])
        # Сохраняем признаки
        self.X = X
        # Сохраняем метки
        self.y = y

    def predict(self, X_test: pd.DataFrame):
        # Список предсказаний
        predictions = []
        for _, test in X_test.iterrows():
            heap = []  # Куча для ближайших соседей
            for train, label in zip(self.X.values, self.y.values):
                distance = self._get_metric(test, train)
                # Заполняем кучу до k элементов
                if len(heap) < self.k:
                    heapq.heappush(heap, (-distance, label))
                else:
                    # Заменяем, если найден ближе
                    if distance < -heap[0][0]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, (-distance, label))
            # Определяем итоговый класс по весам
            pred = self._predict_by_weight(heap)
            predictions.append(1 if pred >= 0.5 else 0)
        return np.array(predictions)

    def _predict_by_weight(self, distances):
        # Список меток соседей
        labels = [label for _, label in distances]
        # Равные веса
        if self.weight == "uniform":
            count_1 = labels.count(1)
            return count_1 / len(labels)
        # Веса по рангу
        if self.weight == "rank":
            sorted_neighbors = sorted(distances, key=lambda x: -x[0])
            k = len(sorted_neighbors)
            weight_1, weight_0 = 0, 0
            for rank, (_, label) in enumerate(sorted_neighbors, start=1):
                w = 1 / rank
                if label == 1:
                    weight_1 += w
                else:
                    weight_0 += w
            return weight_1 / (weight_1 + weight_0)
        # Веса по расстоянию
        if self.weight == "distance":
            score_0, score_1 = 0, 0
            denom = 0
            for neg_d, label in distances:
                d = -neg_d
                if d == 0:
                    return label
                w = 1 / d
                denom += w
                if label == 1:
                    score_1 += w
                else:
                    score_0 += w
            return score_1 / denom

    def predict_proba(self, X_test: pd.DataFrame):
        # Вероятности принадлежности к классу 1
        predictions = []
        for _, test in X_test.iterrows():
            heap = []
            for train, label in zip(self.X.values, self.y.values):
                distance = self._get_metric(test, train)
                if len(heap) < self.k:
                    heapq.heappush(heap, (-distance, label))
                else:
                    if distance < -heap[0][0]:
                        heapq.heappop(heap)
                        heapq.heappush(heap, (-distance, label))
            predictions.append(self._predict_by_weight(heap))
        return np.array(predictions)

    def _get_metric(self, X1: np.array, X2: np.array):
        # Выбор метрики расстояния
        if self.metric == "euclidean":
            return self._get_euclidean(X1, X2)
        if self.metric == "manhattan":
            return self._get_manhattan(X1, X2)
        if self.metric == "chebyshev":
            return self._get_chebyshev(X1, X2)
        if self.metric == "cosine":
            return self._get_cosine(X1, X2)

    def _get_euclidean(self, X1: np.array, X2: np.array):
        # Евклидово расстояние
        return np.sum((X1 - X2) ** 2) ** 0.5

    def _get_manhattan(self, X1: np.array, X2: np.array):
        # Манхэттенское расстояние
        return np.sum(abs(X1 - X2))

    def _get_chebyshev(self, X1: np.array, X2: np.array):
        # Расстояние Чебышева
        return np.max(np.abs(X1 - X2))

    def _get_cosine(self, X1: np.array, X2: np.array):
        # Косинусное расстояние
        return 1 - np.sum(X1 * X2) / (np.sqrt(np.sum(X1 ** 2)) * np.sqrt(np.sum(X2 ** 2)))
