"""
subspace_analysis.py  —  Person 2: Similarity & Vector Space Engine
Analyzes the rating matrix using the Four Fundamental Subspaces theorem.

Linear Algebra concepts:
  - Rank of a matrix: dim(col(A))
  - Nullity: dim(null(A)) = n - rank(A)   [Rank-Nullity Theorem]
  - Four Fundamental Subspaces of A (m x n):
      1. Column space  C(A)   ⊆ R^m        — span of columns
      2. Row space     C(A^T) ⊆ R^n        — span of rows
      3. Null space    N(A)   ⊆ R^n        — {x : Ax = 0}
      4. Left null space N(A^T) ⊆ R^m      — {y : A^T y = 0}

  Rank-Nullity Theorem:
      rank(A) + nullity(A) = n  (number of columns)
"""

import numpy as np
from scipy.linalg import svd, null_space


def matrix_rank(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute the numerical rank of matrix A using SVD.
    rank(A) = number of singular values > tol.
    """
    _, s, _ = svd(A, full_matrices=False)
    return int(np.sum(s > tol))


def nullity(A: np.ndarray, tol: float = 1e-10) -> int:
    """
    Compute nullity of A.
    By Rank-Nullity theorem: nullity(A) = n_cols - rank(A)
    """
    return A.shape[1] - matrix_rank(A, tol)


def column_space_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Basis for the column space C(A).
    The first rank(A) left singular vectors of the SVD form an orthonormal basis.

    A = U Σ V^T  →  C(A) = span of first r columns of U
    """
    U, s, _ = svd(A, full_matrices=True)
    r = int(np.sum(s > tol))
    return U[:, :r]


def row_space_basis(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Basis for the row space C(A^T).
    The first rank(A) right singular vectors form an orthonormal basis.

    A = U Σ V^T  →  C(A^T) = span of first r columns of V
    """
    _, s, Vt = svd(A, full_matrices=True)
    r = int(np.sum(s > tol))
    V = Vt.T
    return V[:, :r]


def null_space_basis(A: np.ndarray) -> np.ndarray:
    """
    Basis for the null space N(A) = {x : Ax = 0}.
    Uses scipy's null_space (SVD-based).
    Returns shape (n_cols, nullity) or empty if nullity=0.
    """
    ns = null_space(A)
    return ns


def left_null_space_basis(A: np.ndarray) -> np.ndarray:
    """
    Basis for the left null space N(A^T) = {y : A^T y = 0}.
    Equivalent to null space of A^T.
    """
    return null_space(A.T)


def rank_nullity_check(A: np.ndarray) -> dict:
    """
    Verify the Rank-Nullity theorem for matrix A.

    Returns a dict with rank, nullity, n_cols, and whether theorem holds.
    """
    r = matrix_rank(A)
    n = nullity(A)
    n_cols = A.shape[1]
    theorem_holds = (r + n == n_cols)

    return {
        "shape": A.shape,
        "rank": r,
        "nullity": n,
        "n_columns": n_cols,
        "rank_nullity_sum": r + n,
        "theorem_holds": theorem_holds,
    }


def analyze_rating_matrix(R: np.ndarray) -> dict:
    """
    Full subspace analysis of the user-restaurant rating matrix.

    Interpretation:
      - rank(R)    : number of independent user preference dimensions
      - nullity(R) : restaurants that cannot be distinguished by any user's ratings
      - Singular values indicate how much 'information' each latent dimension captures
    """
    info = rank_nullity_check(R)

    _, s, _ = svd(R, full_matrices=False)
    # Explained variance (information) per singular value
    s_sq = s ** 2
    total = s_sq.sum()
    explained = (s_sq / total * 100) if total > 0 else np.zeros_like(s)

    info["singular_values"] = s.tolist()
    info["top5_explained_variance_%"] = explained[:5].tolist()
    info["col_space_dim"]  = info["rank"]
    info["row_space_dim"]  = info["rank"]
    info["null_space_dim"] = info["nullity"]
    info["left_null_dim"]  = R.shape[0] - info["rank"]

    return info


if __name__ == "__main__":
    import os, pandas as pd
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from similarity import build_rating_matrix

    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    ratings_df     = pd.read_csv(os.path.join(BASE, "ratings.csv"))
    users_df       = pd.read_csv(os.path.join(BASE, "users.csv"))
    restaurants_df = pd.read_csv(os.path.join(BASE, "restaurants.csv"))

    R, user_ids, rest_ids = build_rating_matrix(ratings_df, users_df, restaurants_df)
    print(f"Rating matrix R: {R.shape}  (users x restaurants)")

    result = analyze_rating_matrix(R)
    print(f"\n=== Four Fundamental Subspaces ===")
    print(f"Matrix shape:       {result['shape']}")
    print(f"Rank:               {result['rank']}  (independent preference dimensions)")
    print(f"Nullity:            {result['nullity']}  (undistinguishable restaurants)")
    print(f"Column space dim:   {result['col_space_dim']}")
    print(f"Row space dim:      {result['row_space_dim']}")
    print(f"Null space dim:     {result['null_space_dim']}")
    print(f"Left null space dim:{result['left_null_dim']}")
    print(f"\nRank-Nullity check: rank + nullity = {result['rank_nullity_sum']} == n_cols={result['n_columns']}  →  {result['theorem_holds']}")
    print(f"\nTop-5 singular values: {[round(s,3) for s in result['singular_values'][:5]]}")
    print(f"Top-5 explained var%:  {[round(v,1) for v in result['top5_explained_variance_%']]}")
