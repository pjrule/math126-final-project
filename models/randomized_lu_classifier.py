"""Randomized LU-based dictionary classification."""
import numpy as np
from numpy.linalg import pinv
from typing import Optional, Dict, Any
from tqdm import tqdm
from .dict_classifier import DictClassifier
from .randomized_lu import randomized_lu


class RandomizedLUClassifier(DictClassifier):
    """Randomized LU-based dictionary classifier."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomizedLUClassifier':
        """Trains a dictionary for each dataset in X."""
        self.dictionary_invs = {}
        self.classes = np.unique(y)

        classes = self.classes if self.verbosity == 0 else tqdm(self.classes)
        for class_ in classes:
            X_class = X[np.where(y == class_)[0]]
            P, Q, L, U = randomized_lu(X_class.T, self.k, self.k + 5,
                                       self.random_state)
            D = P.T @ L
            self.dictionary_invs[class_] = D @ pinv(D)
        self._fit_error_model(X, y)
        return self
