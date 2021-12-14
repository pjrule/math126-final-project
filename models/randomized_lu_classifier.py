"""Randomized LU-based classifier."""
import numpy as np
from numpy.linalg import pinv, norm
from sklearn.base import BaseEstimator, ClassifierMixin
from randomized_lu import randomized_lu


class RandomizedLUClassifier(BaseEstimator, ClassifierMixin):
    """Randomized LU-based classifier."""
    def dist(self, x: np.ndarray, D: np.ndarray) -> float:
        """Calculates the distance metric between an input
           vector x and a dictionary D

        Args:
            x: The input vector.
            D: A trained dictionary multiplied by its pseudoinverse.

        Returns:
            The distance between x and D."""
        return norm(D @ x - x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts which class the input vector x belongs to
           given a list of dictionaries

        Args:
            X: Matrix of input vectors to classify

        Returns:
            The predicted class of each vector in X."""

        class_predictions = [
            self.classes[np.argmin([
                self.dist(x, self.dictionary_invs[class_])
                for class_ in self.classes
            ])] for x in X
        ]
        return class_predictions

    def fit(self, X: np.ndarray, y: np.ndarray, k: dict = {}) -> list:
        """Trains a dictionary of size k for each dataset in X

        Args:
            X: A matrix of input vectors.
            y: A vector of class labels.
            k: A map for the size of each class's dictionary. (key,val) = (class, size)

        Returns:
            A DictLearner object with a trained dictionary for each class"""
        self.dictionaries = {}
        self.dictionary_invs = {}
        self.classes = np.unique(y)

        for class_ in self.classes:
            X_class = X[np.where(y == class_)[0]]
            P, Q, L, U = randomized_lu(X_class.T, k[class_], k[class_] + 5)
            D = P.T @ L
            self.dictionaries[class_] = D
            self.dictionary_invs[class_] = D @ pinv(D)
        return self
