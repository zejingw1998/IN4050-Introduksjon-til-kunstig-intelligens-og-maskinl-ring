from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import math

import numpy as np


class DistanceStrategy(ABC):
    @abstractmethod
    def distance(self, a, b):
        """Return a distance between two feature vectors."""


class L1Distance(DistanceStrategy):
    def distance(self, a, b):
        return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


class L2Distance(DistanceStrategy):
    def distance(self, a, b):
        return float(np.sqrt(np.sum((np.asarray(a) - np.asarray(b)) ** 2)))


class SquaredL2Distance(DistanceStrategy):
    def distance(self, a, b):
        return float(np.sum((np.asarray(a) - np.asarray(b)) ** 2))


class VotingStrategy(ABC):
    @abstractmethod
    def vote(self, neighbor_labels, neighbor_distances, classes):
        """Return (prediction, probabilities) from neighbor labels and distances."""


class MajorityVoting(VotingStrategy):
    def vote(self, neighbor_labels, neighbor_distances, classes):
        counts = Counter(neighbor_labels)
        prediction = min(counts, key=lambda label: (-counts[label], label))
        total = len(neighbor_labels)
        probabilities = {label: counts.get(label, 0) / total for label in classes}
        return prediction, probabilities


class ProbabilityVoting(MajorityVoting):
    pass


class DistanceWeightedVoting(VotingStrategy):
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def vote(self, neighbor_labels, neighbor_distances, classes):
        scores = defaultdict(float)
        for label, distance in zip(neighbor_labels, neighbor_distances):
            scores[label] += 1.0 / (float(distance) + self.epsilon)

        total = sum(scores.values())
        probabilities = {label: scores.get(label, 0.0) / total for label in classes}
        prediction = min(probabilities, key=lambda label: (-probabilities[label], label))
        return prediction, probabilities


class KNNClassifier:
    def __init__(self, k=3, distance_strategy=None, voting_strategy=None):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self.distance_strategy = distance_strategy or SquaredL2Distance()
        self.voting_strategy = voting_strategy or MajorityVoting()

    def fit(self, X, y):
        self.X_train = [np.asarray(row) for row in X]
        self.y_train = list(y)
        self.classes_ = sorted(set(self.y_train))
        if self.k > len(self.X_train):
            raise ValueError("k cannot be larger than the training set")
        return self

    def _neighbors(self, x):
        distances = [
            self.distance_strategy.distance(x, train_x)
            for train_x in self.X_train
        ]
        order = sorted(range(len(distances)), key=lambda i: distances[i])[: self.k]
        labels = [self.y_train[i] for i in order]
        neighbor_distances = [distances[i] for i in order]
        return order, labels, neighbor_distances

    def predict_one(self, x):
        _, labels, distances = self._neighbors(x)
        prediction, _ = self.voting_strategy.vote(labels, distances, self.classes_)
        return prediction

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def predict_proba_one(self, x):
        _, labels, distances = self._neighbors(x)
        _, probabilities = self.voting_strategy.vote(labels, distances, self.classes_)
        return probabilities

    def predict_proba(self, X):
        return [self.predict_proba_one(x) for x in X]


class FastKNNClassifier:
    def __init__(self, k=3, distance="squared_l2", voting="majority", epsilon=1e-8):
        self.k = k
        self.distance = distance
        self.voting = voting
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y)
        self.classes_ = np.array(sorted(set(self.y_train.tolist())))
        if self.k > len(self.X_train):
            raise ValueError("k cannot be larger than the training set")
        return self

    def _distances(self, x):
        x = np.asarray(x, dtype=np.float32)
        diff = self.X_train - x
        if self.distance == "l1":
            return np.sum(np.abs(diff), axis=1)
        if self.distance == "l2":
            return np.sqrt(np.sum(diff ** 2, axis=1))
        if self.distance == "squared_l2":
            return np.sum(diff ** 2, axis=1)
        raise ValueError(f"Unknown distance: {self.distance}")

    def _neighbors(self, x):
        distances = self._distances(x)
        idx = np.argpartition(distances, self.k - 1)[: self.k]
        idx = idx[np.argsort(distances[idx])]
        return idx, self.y_train[idx], distances[idx]

    def predict_proba_one(self, x):
        _, labels, distances = self._neighbors(x)
        scores = {label: 0.0 for label in self.classes_}

        if self.voting == "majority":
            for label in labels:
                scores[label] += 1.0
        elif self.voting == "distance_weighted":
            for label, distance in zip(labels, distances):
                scores[label] += 1.0 / (float(distance) + self.epsilon)
        else:
            raise ValueError(f"Unknown voting: {self.voting}")

        total = sum(scores.values())
        return {label: score / total for label, score in scores.items()}

    def predict_one(self, x):
        probabilities = self.predict_proba_one(x)
        return min(probabilities, key=lambda label: (-probabilities[label], label))

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])


def accuracy_score_simple(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))
