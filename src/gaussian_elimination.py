"""Gaussian elimination and LU decomposition routines."""

"""
gaussian_elimination.py — LU Decomposition & Linear System Solver
==================================================================
Person 1 | TasteVector Project

Implements LU Decomposition from scratch using only NumPy arrays.
Does NOT call scipy.linalg.lu — every step is hand-coded.

Factorization:  A = L @ U
  L : lower triangular matrix (with 1s on diagonal)
  U : upper triangular matrix

Solving Ax = b in two passes:
  1. Forward substitution : solve  L y = b  for y
  2. Back substitution    : solve  U x = y  for x

Application in TasteVector
--------------------------
Given a user's sparse preference vector b (ratings for a few restaurants)
and the feature sub-matrix A for those rated restaurants:

    A x = b   →   solve for x = feature weight vector

x tells us how much the user cares about price, spice, distance, etc.
That weight vector is then dotted with every unrated restaurant's
feature vector to produce a predicted score.
"""

import numpy as np


# ── LU Decomposition ─────────────────────────────────────────────────────────

def lu_decompose(A: np.ndarray) -> tuple:
    """
    Factor a square matrix A into A = L @ U using Gaussian Elimination
    with partial pivoting.

    Partial pivoting: at each step, swap the current row with the row
    that has the largest absolute value in the pivot column. This improves
    numerical stability and avoids division by zero.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)  — must be square and non-singular

    Returns
    -------
    L : (n, n) lower triangular, diagonal entries = 1
    U : (n, n) upper triangular
    P : (n, n) permutation matrix  (tracks row swaps)
               satisfies  P @ A = L @ U

    Raises
    ------
    ValueError  if A is not square
    np.linalg.LinAlgError  if A is singular (pivot becomes 0)
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"lu_decompose requires a square matrix, got {A.shape}")

    L = np.eye(n)           # start as identity; filled column by column
    U = A.copy()            # will be reduced to upper triangular in place
    P = np.eye(n)           # permutation matrix

    for col in range(n):
        # ── Partial pivoting ──────────────────────────────────────────────
        max_row = col + np.argmax(np.abs(U[col:, col]))
        if max_row != col:
            U[[col, max_row]] = U[[max_row, col]]
            P[[col, max_row]] = P[[max_row, col]]
            if col > 0:
                L[[col, max_row], :col] = L[[max_row, col], :col]

        pivot = U[col, col]
        if abs(pivot) < 1e-12:
            raise np.linalg.LinAlgError(
                f"Zero pivot encountered at column {col} — matrix is singular."
            )

        # ── Elimination: zero out entries below the pivot ─────────────────
        for row in range(col + 1, n):
            factor = U[row, col] / pivot
            L[row, col] = factor
            U[row, col:] -= factor * U[col, col:]

    return L, U, P


# ── Forward & back substitution ──────────────────────────────────────────────

def forward_substitution(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve  L y = b  for y  where L is lower triangular.

    y_i = (b_i - sum_{j<i} L[i,j] * y[j]) / L[i,i]

    Since L has 1s on the diagonal (from LU decomp), L[i,i] = 1 always.
    """
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    return y


def back_substitution(U: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Solve  U x = y  for x  where U is upper triangular.

    x_i = (y_i - sum_{j>i} U[i,j] * x[j]) / U[i,i]
    """
    n = len(y)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]
    return x


# ── Combined solver ──────────────────────────────────────────────────────────

def solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the linear system  A x = b  using LU Decomposition.

    Steps
    -----
    1. Decompose:  P A = L U
    2. Apply permutation:  b' = P b
    3. Forward sub:  L y = b'   →   y
    4. Back sub:     U x = y    →   x

    Parameters
    ----------
    A : (n, n) square non-singular matrix
    b : (n,)   right-hand side vector

    Returns
    -------
    x : (n,) solution vector such that A @ x ≈ b
    """
    b = np.array(b, dtype=float)
    L, U, P = lu_decompose(A)
    b_perm = P @ b              # apply the same row permutations to b
    y = forward_substitution(L, b_perm)
    x = back_substitution(U, y)
    return x


# ── Application: user preference weights ─────────────────────────────────────

def solve_preference_weights(F_rated: np.ndarray,
                             ratings_vec: np.ndarray) -> np.ndarray:
    """
    Given a user's ratings for a subset of restaurants and the feature
    vectors of those restaurants, solve for the user's feature weight vector.

    System:  F_rated @ x = ratings_vec
             (n_rated x n_features) @ (n_features,) = (n_rated,)

    When n_rated != n_features this is over/under-determined.
    We solve the Normal Equations instead:
             F^T F x = F^T b   (Least Squares)

    Parameters
    ----------
    F_rated     : (n_rated, n_features)  feature matrix for rated restaurants
    ratings_vec : (n_rated,)             user's actual ratings

    Returns
    -------
    x : (n_features,) weight vector  [w_price, w_spice, w_distance, w_veg]
    """
    A = F_rated.T @ F_rated      # (n_features, n_features)  — always square
    b = F_rated.T @ ratings_vec  # (n_features,)
    return solve(A, b)


def score_restaurants(F: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Dot every restaurant's feature vector with the user's weight vector.

    score_j = F[j] · weights

    Returns
    -------
    scores : (n_restaurants,)  predicted preference score per restaurant
    """
    return F @ weights


# ── Verification helpers (used in tests) ─────────────────────────────────────

def verify_lu(A: np.ndarray, L: np.ndarray,
              U: np.ndarray, P: np.ndarray,
              tol: float = 1e-8) -> bool:
    """Check that P @ A ≈ L @ U within floating-point tolerance."""
    return bool(np.allclose(P @ A, L @ U, atol=tol))


def residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """||A x - b||  — should be near zero for a correct solution."""
    return float(np.linalg.norm(A @ x - b))


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "src"))
    from data_loader    import load_all
    from matrix_builder import build_feature_matrix, build_rating_matrix

    print("=" * 55)
    print("LU DECOMPOSITION TESTS")
    print("=" * 55)

    test_systems = [
        # (A, b, label)
        (
            np.array([[2., 1., -1.], [-3., -1., 2.], [-2., 1., 2.]]),
            np.array([8., -11., -3.]),
            "3x3 classic"
        ),
        (
            np.array([[1., 2., 0., 0.], [3., 4., 2., 0.],
                      [0., 1., 3., 1.], [0., 0., 2., 4.]]),
            np.array([1., 2., 3., 4.]),
            "4x4 banded"
        ),
        (
            np.array([[4., 3.], [6., 3.]]),
            np.array([10., 12.]),
            "2x2 simple"
        ),
    ]

    for A, b, label in test_systems:
        L, U, P = lu_decompose(A)
        x = solve(A, b)
        lu_ok  = verify_lu(A, L, U, P)
        res    = residual(A, x, b)
        x_ref  = np.linalg.solve(A, b)
        print(f"\n[{label}]")
        print(f"  x (ours)     : {np.round(x, 6)}")
        print(f"  x (np.linalg): {np.round(x_ref, 6)}")
        print(f"  LU verified  : {lu_ok}")
        print(f"  Residual ||Ax-b||: {res:.2e}")

    print("\n" + "=" * 55)
    print("PREFERENCE WEIGHT SOLVER")
    print("=" * 55)

    restaurants, users, ratings = load_all(os.path.join(base, "data"))
    F = build_feature_matrix(restaurants)
    R = build_rating_matrix(users, restaurants, ratings)

    # User 0 (Alice) — use her rated restaurants
    user_idx = 0
    rated_mask   = R[user_idx] != 0
    F_rated      = F[rated_mask]
    ratings_vec  = R[user_idx, rated_mask]

    weights = solve_preference_weights(F_rated, ratings_vec)
    print(f"\nAlice's feature weights: {dict(zip(['price','spice','dist','veg'], np.round(weights,4)))}")

    scores = score_restaurants(F, weights)
    top5   = np.argsort(scores)[::-1][:5]
    print("\nTop-5 restaurants by weight-based score:")
    for idx in top5:
        name = restaurants.iloc[idx]["name"]
        print(f"  {name:<25}  score={scores[idx]:.3f}")
