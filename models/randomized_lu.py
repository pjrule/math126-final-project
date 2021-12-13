"""Randomized LU decomposition."""
import numpy as np
import scipy.linalg as la
from typing import Tuple

PQLU = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

def randomized_lu(A: np.ndarray, k: int, l: int, seed: int=0) -> PQLU:
    """Performs a randomized rank-k LU decomposition of A.
    
    Adapted from Shabat et al. 2013, Algorithm 4.1.

    Args:
        A: An mXn matrix to decompose.
        k: Rank of the decomposition.
        l: Number of columns to use in the random matrix.
        seed: Random seed.

    Returns:
        A 4-tuple containing P, Q, L, U."""
    rand = np.random.RandomState(seed)
    # 1. Create a matrix G of size n × l whose entries are i.i.d. Gaussian
    # random variables with zero mean and unit standard deviation.
    assert l >= k
    m, n = A.shape
    G = rand.randn(n, l)

    # 2. Y ← AG.    
    Y = A @ G  
    assert Y.shape == (m, l)

    # 3. Apply RRLU decomposition (Theorem 3.1) to Y such that P Y Qy = LyUy.
    #
    # Remark 4.2. In practice, it is sufficient to perform step 3 in Algorithm
    # 4.1 using standard LU decomposition with partial pivoting instead of
    # applying RRLU. The cases where U grows exponentially are extremely rare...
    P, L_y, U_y = la.lu(Y)
    P = P.T
    Q_y = np.identity(l) # TODO: replace with RRLU
    assert P.shape == (m, m)
    assert L_y.shape == (m, l)
    assert U_y.shape == (l, l)
    #assert np.allclose(P @ Y, L_y @ U_y)
    #assert np.allclose(P @ Y @ Q_y, L_y @ U_y)

    # 4. Truncate Ly and Uy by choosing the first k columns and the first k rows,
    # respectively, such that Ly ← Ly(:, 1 : k) and Uy ← Uy(1 : k, :).
    L_y = L_y[:, :k]
    U_y = U_y[:k, :]
    assert L_y.shape == (m, k) 
    assert U_y.shape == (k, l)

    # 5. B ← (L_y †) PA
    L_y_pseudoinverse = la.pinv(L_y)
    assert L_y_pseudoinverse.shape == (k, m)
    B = L_y_pseudoinverse @ P @ A
    assert B.shape == (k, n)

    # 6. Apply LU decomposition to B with column pivoting BQ = L_b U_b.
    Q, U_b, L_b = la.lu(B.T)
    #Q = Q.T
    L_b = L_b.T
    U_b = U_b.T
    assert Q.shape == (n, n)
    assert L_b.shape == (k, k)
    assert U_b.shape == (k, n)
    #assert np.allclose(B @ Q, L_b @ U_b)

    # 7. L ← L_y L_b.
    L = L_y @ L_b
    assert L.shape == (m, k)

    return P, Q, L, U_b