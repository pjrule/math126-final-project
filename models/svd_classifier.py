"""SVD-based dictionary classification."""
import numpy as np
from numpy.linalg import pinv
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from .dict_classifier import DictClassifier


class SVDClassifier(DictClassifier):
    """SVD-based dictionary classifier."""
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVDClassifier':
        """Trains a dictionary for each dataset in X."""
        self.dictionary_invs = {}
        self.classes = np.unique(y)

        classes = self.classes if self.verbosity == 0 else tqdm(self.classes)
        for class_ in classes:
            X_class = X[np.where(y == class_)[0]]
            transformer = TruncatedSVD(n_components=self.k,
                                       algorithm='randomized',
                                       random_state=self.random_state)
            transformer.fit(X_class)
            D = transformer.components_
            self.dictionary_invs[class_] = D @ pinv(D)
        return self
