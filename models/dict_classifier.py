"""Generic undercomplete dictionary classification."""
import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin


def dist(self, x: np.ndarray, D: np.ndarray) -> float:
    """Calculates the distance metric between an input
        vector x and a dictionary D

    Args:
        x: The input vector.
        D: A trained dictionary multiplied by its pseudoinverse.

    Returns:
        The distance between x and D."""
    return norm(D @ x - x)


class DictClassifier(BaseEstimator, ClassifierMixin):
    """Generic undercomplete dictionary-based classification."""
    def __init__(self, k: int, random_state: int, verbosity: int = 0):
        """
        Args:
            k: Rank of each class' dictionary.
            random_state: Seed for randomized decomposition operations.
            verbosity: Verbosity level (enables per-class progress bar when >0).
        """
        self.k = k
        self.random_state = random_state
        self.verbosity = verbosity

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
