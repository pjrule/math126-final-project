"""Generic undercomplete dictionary classification."""
import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def dist(x: np.ndarray, D: np.ndarray) -> float:
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
    def __init__(self,
                 k: int,
                 random_state: int,
                 error_classifier: str = 'xgboost',
                 verbosity: int = 0):
        """
        Args:
            k: Rank of each class' dictionary.
            random_state: Seed for randomized decomposition operations.
            error_classifier: The type of classifier to use for classifying
              on reconstruction error. (Options are `nearest`, `svm`, and `xgboost`).
            verbosity: Verbosity level (enables per-class progress bar when >0).
        """
        self.k = k
        self.random_state = random_state
        self.error_classifier = error_classifier
        self.verbosity = verbosity

    def _fit_error_model(self, X: np.ndarray, y: np.ndarray):
        """Fits an internal model based on reconstruction error."""
        reconstruction_errors = np.zeros(
            (len(self.dictionary_invs), X.shape[0]))
        for dict_idx, dictionary in self.dictionary_invs.items():
            for sample_idx, sample in enumerate(X):
                sample_norm = np.linalg.norm(sample)
                if sample_norm > 0:
                    x_normed = sample / np.linalg.norm(sample)
                reconstruction_errors[dict_idx, sample_idx] = np.linalg.norm(
                    dictionary @ x_normed - x_normed)

        if self.error_classifier == 'xgboost':
            self.error_model = LinearSVC()
        elif self.error_classifier == 'svc':
            self.error_model = XGBClassifier()
        elif self.error_classifier == 'nearest':
            self.error_model = None
            return
        else:
            raise ValueError(
                f'Unknown error classifier {self.error_classifier}.')

        self.error_model.fit(reconstruction_errors.T, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts which class the input vector x belongs to
           given a list of dictionaries

        Args:
            X: Matrix of input vectors to classify

        Returns:
            The predicted class of each vector in X."""
        dists = [[
            dist(x, self.dictionary_invs[class_]) for class_ in self.classes
        ] for x in X]

        if self.error_model is None:
            return np.argmin(dists, axis=1)
        return self.error_model.predict(dists)
