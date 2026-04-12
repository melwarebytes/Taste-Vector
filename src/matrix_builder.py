"""Matrix construction utilities for TasteVector."""

"""
matrix_builder.py — Rating & Feature Matrix Construction
=========================================================
Person 1 | TasteVector Project

Builds the two core NumPy matrices used throughout the pipeline:

  R : (n_users x n_restaurants)   user-restaurant rating matrix
                                   R[i][j] = rating user i gave restaurant j
                                   0 means "not rated" (sparse)

  F : (n_restaurants x n_features) restaurant feature matrix
                                   each row = [price, spice, distance_km, veg_friendly]

Also implements:
  - Matrix operations (addition, multiplication, transpose, inversion)
  - Mean-centering of the rating matrix
  - LU Decomposition via Gaussian Elimination  ← see gaussian_elimination.py
"""

import numpy as np
import pandas as pd


# ── Feature columns used to build F ─────────────────────────────────────────
FEATURE_COLS = ["price", "spice", "distance_km", "veg_friendly"]


# ── Rating matrix ────────────────────────────────────────────────────────────

def build_rating_matrix(users: pd.DataFrame,
                        restaurants: pd.DataFrame,
                        ratings: pd.DataFrame) -> np.ndarray:
    """
    Construct the user-restaurant rating matrix R.

    R[i][j] = rating that user i gave restaurant j
    R[i][j] = 0  if user i has not rated restaurant j  (sparse / missing)

    Parameters
    ----------
    users       : users DataFrame (must have 'user_id' column)
    restaurants : restaurants DataFrame (must have 'restaurant_id' column)
    ratings     : ratings DataFrame (user_id, restaurant_id, rating)

    Returns
    -------
    R : np.ndarray, shape (n_users, n_restaurants), dtype float64
    """
    n_users = len(users)
    n_rest  = len(restaurants)

    # Build index maps: id → row/col position
    uid_to_idx = {uid: i for i, uid in enumerate(users["user_id"])}
    rid_to_idx = {rid: j for j, rid in enumerate(restaurants["restaurant_id"])}

    R = np.zeros((n_users, n_rest), dtype=float)

    for _, row in ratings.iterrows():
        i = uid_to_idx.get(row["user_id"])
        j = rid_to_idx.get(row["restaurant_id"])
        if i is not None and j is not None:
            R[i, j] = float(row["rating"])

    return R


# ── Feature matrix ────────────────────────────────────────────────────────────

def build_feature_matrix(restaurants: pd.DataFrame) -> np.ndarray:
    """
    Construct the restaurant feature matrix F.

    Each row represents one restaurant as a numeric vector:
        [price, spice, distance_km, veg_friendly]

    Parameters
    ----------
    restaurants : restaurants DataFrame

    Returns
    -------
    F : np.ndarray, shape (n_restaurants, n_features), dtype float64
    """
    return restaurants[FEATURE_COLS].to_numpy(dtype=float)


# ── Matrix operations ────────────────────────────────────────────────────────

def mat_add(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Element-wise matrix addition.  A and B must have the same shape."""
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: {A.shape} vs {B.shape}")
    return A + B


def mat_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication  C = A @ B.
    A : (m, k)   B : (k, n)   →   C : (m, n)
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Incompatible shapes for multiplication: {A.shape} @ {B.shape}"
        )
    return A @ B


def mat_transpose(A: np.ndarray) -> np.ndarray:
    """Transpose of A."""
    return A.T


def mat_inverse(A: np.ndarray) -> np.ndarray:
    """
    Inverse of a square matrix A.
    Raises np.linalg.LinAlgError if A is singular.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square to invert, got {A.shape}")
    return np.linalg.inv(A)


# ── Mean-centering ────────────────────────────────────────────────────────────

def mean_center(R: np.ndarray) -> tuple:
    """
    Subtract each user's mean rating from their rated entries.

    For user i:
        mean_i  = average of all non-zero entries in R[i]
        R_c[i, j] = R[i, j] - mean_i   for every j where R[i, j] != 0
        R_c[i, j] = 0                   for unrated entries (unchanged)

    This corrects for users who consistently rate high or low ("bias").

    Returns
    -------
    R_centered : np.ndarray, same shape as R
    user_means : np.ndarray, shape (n_users,)
    """
    R_centered = R.astype(float).copy()
    user_means = np.zeros(R.shape[0])

    for i in range(R.shape[0]):
        rated_mask = R[i] != 0
        if rated_mask.any():
            mu = R[i, rated_mask].mean()
            user_means[i] = mu
            R_centered[i, rated_mask] -= mu

    return R_centered, user_means


def restore_means(R_approx: np.ndarray, user_means: np.ndarray) -> np.ndarray:
    """Add per-user means back (undo mean-centering)."""
    return R_approx + user_means[:, np.newaxis]


# ── Sparsity analysis ─────────────────────────────────────────────────────────

def sparsity_report(R: np.ndarray) -> dict:
    """
    Return a dict with basic sparsity stats about R.
    Used in subspace_analysis.py and tests.
    """
    total   = R.size
    rated   = int(np.count_nonzero(R))
    unrated = total - rated
    return {
        "shape":          R.shape,
        "total_cells":    total,
        "rated":          rated,
        "unrated":        unrated,
        "sparsity_pct":   round(100.0 * unrated / total, 2),
        "density_pct":    round(100.0 * rated   / total, 2),
    }


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "src"))
    from data_loader import load_all

    restaurants, users, ratings = load_all(os.path.join(base, "data"))

    R = build_rating_matrix(users, restaurants, ratings)
    F = build_feature_matrix(restaurants)

    print(f"Rating matrix R  : {R.shape}")
    print(f"Feature matrix F : {F.shape}")

    stats = sparsity_report(R)
    print(f"\nSparsity report:")
    for k, v in stats.items():
        print(f"  {k:<18}: {v}")

    R_c, means = mean_center(R)
    print(f"\nUser means (per-row average of rated entries):")
    for i, mu in enumerate(means):
        name = users.iloc[i]["name"]
        print(f"  {name:<10}: {mu:.3f}")

    # Verify mean-centering: mean of centered rated entries should be ~0
    for i in range(R.shape[0]):
        mask = R[i] != 0
        if mask.any():
            assert abs(R_c[i, mask].mean()) < 1e-10, f"User {i} not centered!"
    print("\nMean-centering verified: all per-user means ≈ 0 ✓")

    # Basic matrix ops
    print("\nMatrix operation checks:")
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    print(f"  A + B =\n{mat_add(A, B)}")
    print(f"  A @ B =\n{mat_multiply(A, B)}")
    print(f"  A^T   =\n{mat_transpose(A)}")
    print(f"  A^-1  =\n{np.round(mat_inverse(A), 4)}")
