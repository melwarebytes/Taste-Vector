"""
projection.py  —  Person 2: Similarity & Vector Space Engine
Least squares projections and subspace projections for new/sparse users.

Linear Algebra concepts:
  - Orthogonal projection onto a subspace: P = Q Q^T  (if Q has orthonormal columns)
  - Least squares: given Ax ≈ b, the normal equations give x* = (A^T A)^{-1} A^T b
  - Projection of a new user vector onto the column space of the rating matrix
  - Pseudo-inverse: A^+ = (A^T A)^{-1} A^T  (Moore-Penrose)
"""

import numpy as np


def projection_matrix(Q: np.ndarray) -> np.ndarray:
    """
    Build the orthogonal projection matrix onto the column space of Q.

    If Q has orthonormal columns (Q^T Q = I):
        P = Q Q^T

    If Q is arbitrary:
        P = Q (Q^T Q)^{-1} Q^T  (general formula)

    Parameters:
        Q : (d, k) matrix whose columns span the subspace

    Returns:
        P : (d, d) projection matrix,  P^2 = P,  P^T = P
    """
    QtQ = Q.T @ Q
    # Check if columns are already orthonormal
    if np.allclose(QtQ, np.eye(Q.shape[1]), atol=1e-8):
        return Q @ Q.T
    else:
        return Q @ np.linalg.solve(QtQ, Q.T)


def project_onto_subspace(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Project vector v onto the subspace spanned by columns of Q.

    proj = P v  where  P = Q (Q^T Q)^{-1} Q^T

    Parameters:
        v : (d,) vector to project
        Q : (d, k) basis matrix

    Returns:
        v_proj : (d,) projection of v onto col(Q)
    """
    P = projection_matrix(Q)
    return P @ v


def least_squares_preference(R: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Fit user preference weights using least squares.

    We want to represent each user's rating vector r_i ≈ Q w_i,
    where Q is our orthonormal feature basis and w_i is the weight vector.

    Least squares solution:
        w_i* = (Q^T Q)^{-1} Q^T r_i = Q^T r_i  (when Q is orthonormal)

    Parameters:
        R : (n_users, d) rating matrix
        Q : (k, d) orthonormal basis matrix (rows are basis vectors)

    Returns:
        W : (n_users, k) weight matrix — row i is user i's preference weights
    """
    # Q rows are basis vectors, so Q is (k, d)
    # We need Q^T of shape (d, k) to project each user vector (d,)
    W = R @ Q.T  # (n_users, d) @ (d, k) = (n_users, k)
    return W


def predict_new_user_ratings(new_user_vec: np.ndarray,
                              Q: np.ndarray,
                              R: np.ndarray) -> np.ndarray:
    """
    Given a new/sparse user with partial ratings (new_user_vec),
    project them into the preference subspace spanned by Q,
    then predict their full rating vector.

    Steps:
        1. Project new user vector onto col(Q^T): w = Q new_user_vec^T
        2. Reconstruct: r_pred = Q^T w = Q^T Q new_user_vec^T

    Parameters:
        new_user_vec : (d,) partial rating vector (zeros for unrated)
        Q            : (k, d) orthonormal basis matrix
        R            : (n_users, d) existing rating matrix (unused here, kept for API symmetry)

    Returns:
        r_pred : (d,) predicted full rating vector
    """
    w = Q @ new_user_vec          # (k,)  — coordinates in preference subspace
    r_pred = Q.T @ w              # (d,)  — reconstructed ratings
    return r_pred


def reconstruction_error(v: np.ndarray, Q: np.ndarray) -> float:
    """
    Compute the reconstruction error ||v - P v||_2 for a projection.
    A small error means v lies nearly within the subspace.
    """
    v_proj = project_onto_subspace(v, Q.T)  # Q is (k,d), need (d,k) for projection_matrix
    return float(np.linalg.norm(v - v_proj))


if __name__ == "__main__":
    import os, pandas as pd
    from similarity import build_rating_matrix
    from gram_schmidt import gram_schmidt

    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    ratings_df     = pd.read_csv(os.path.join(BASE, "ratings.csv"))
    users_df       = pd.read_csv(os.path.join(BASE, "users.csv"))
    restaurants_df = pd.read_csv(os.path.join(BASE, "restaurants.csv"))

    R, user_ids, rest_ids = build_rating_matrix(ratings_df, users_df, restaurants_df)

    # Build basis from non-zero user vectors
    nonzero_rows = R[np.any(R != 0, axis=1)]
    Q = gram_schmidt(nonzero_rows)  # (k, n_restaurants)

    print(f"Rating matrix R: {R.shape}")
    print(f"Orthonormal basis Q: {Q.shape}")

    # Least squares weights
    W = least_squares_preference(R, Q)
    print(f"User preference weights W: {W.shape}")

    # Simulate a new sparse user: only rated 3 restaurants
    new_user = np.zeros(R.shape[1])
    new_user[0] = 4.0
    new_user[3] = 3.5
    new_user[7] = 2.0

    r_pred = predict_new_user_ratings(new_user, Q, R)
    top_recs = np.argsort(-r_pred)[:5]
    print(f"\nPredicted top-5 restaurants for sparse user:")
    for idx in top_recs:
        print(f"  {rest_ids[idx]}  predicted_rating={r_pred[idx]:.3f}")
