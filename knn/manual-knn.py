import numpy as np
from collections import Counter, defaultdict

class NearestNeighborClassifier:
    def __init__(self, k: int):
        self.k: int = k
        self.X_train: list[list[float]] = []
        self.y_train: list[float] = []
    
    @staticmethod
    def euclidean_distance(a: list[float], b: list[float]) -> float:
        return np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))
    
    def fit(self, X: list[list[float]], y: list[float]) -> None:
        self.X_train: np.ndarray = np.array(X)
        self.y_train: list[int] = y

    def predict(self, point: list[float]) -> int:
        neighbors: np.ndarray = np.array([
            [self.euclidean_distance(point, x), y] 
            for x, y in zip(self.X_train, self.y_train)
        ])

        sorted_neighbors: np.ndarray = neighbors[neighbors[:, 0].argsort()][:self.k]
        votes: Counter[int] = Counter(sorted_neighbors[:, 1])
        sorted_counts: list[int] = sorted(votes.values(), reverse=True)

        if len(sorted_counts) == 1:
            return sorted_neighbors[0][1]
        if sorted_counts[0] != sorted_counts[1]:
            return votes.most_common(1)[0][0]
        if sorted_neighbors[0][0] == 0:
            return sorted_neighbors[0][1]
        
        weights: defaultdict[int, float] = defaultdict(int)
        for distance, label in sorted_neighbors:
            weights[label] += 1 / distance
        
        return max(dict(weights), key=lambda x:weights[x])

class NearestNeighborRegressor(NearestNeighborClassifier):
    def predict(self, point: list[float]) -> float:
        neighbors: np.ndarray = np.array([
            [self.euclidean_distance(point, x), y] 
            for x, y in zip(self.X_train, self.y_train)
        ])

        sorted_neighbors: np.ndarray = neighbors[neighbors[:, 0].argsort()][:self.k]
        return np.mean(sorted_neighbors[:, 1])

knn = NearestNeighborClassifier(k=4)
x1 = [4, 5, 2, 4]
x2 = [21, 19, 60, 17]
X = list(zip(x1, x2))
y = [1, 0, 1, 0]

knn.fit(X, y)
new_point = [8, 21]
prediction = knn.predict(new_point)
print(prediction)