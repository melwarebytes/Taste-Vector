"""
gram_schmidt.py  —  Person 2: Similarity & Vector Space Engine
Gram-Schmidt orthogonalization to build an orthonormal basis for the feature space.

Linear Algebra concepts:
  - Orthogonal projection: proj_u(v) = (<v, u> / <u, u>) * u
  - Gram-Schmidt process: given vectors v_1 ... v_k, produces orthonormal e_1 ... e_k
  - Orthonormal basis: e_i · e_j = delta_ij (identity inner products)
"""

import numpy as np


def gram_schmidt(vectors: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Classical Gram-Schmidt orthogonalization.

    Given a matrix `vectors` of shape (n, d) where each row is a vector in R^d,
    returns a matrix Q of shape (k, d) whose rows form an orthonormal set,
    where k <= n is the number of linearly independent input vectors.

    Algorithm:
      e_1 = v_1 / ||v_1||
      For i = 2, ..., n:
          u_i = v_i - sum_{j<i} proj_{e_j}(v_i)
              = v_i - sum_{j<i} <v_i, e_j> * e_j
          if ||u_i|| > tol:
              e_i = u_i / ||u_i||

    Parameters:
        vectors : (n, d) array — input vectors as rows
        tol     : tolerance below which a vector is considered linearly dependent

    Returns:
        Q : (k, d) orthonormal basis matrix
    """
    basis = []
    for v in vectors:
        # Subtract projections onto all current basis vectors
        u = v.copy().astype(float)
        for e in basis:
            u -= np.dot(u, e) * e  # orthogonal projection subtracted

        norm = np.linalg.norm(u)
        if norm > tol:
            basis.append(u / norm)  # normalize to unit vector

    if len(basis) == 0:
        return np.empty((0, vectors.shape[1]))
    return np.array(basis)


def verify_orthonormality(Q: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Verify Q Q^T ≈ I (orthonormality condition).
    Returns True if all off-diagonal elements < tol and diagonals ≈ 1.
    """
    G = Q @ Q.T  # Gram matrix — should be identity for orthonormal basis
    return bool(np.allclose(G, np.eye(Q.shape[0]), atol=tol))


def restaurant_feature_basis(restaurants_df) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an orthonormal basis from restaurant feature vectors.
    Features used: price, spice, distance_km  (all numeric, continuous)

    Returns:
        feature_matrix : (n_restaurants, 3) raw feature vectors
        Q              : orthonormal basis vectors spanning the feature space
    """
    features = restaurants_df[["price", "spice", "distance_km"]].values.astype(float)

    # Mean-center each feature dimension (standard preprocessing)
    features -= features.mean(axis=0)

    Q = gram_schmidt(features)
    return features, Q


if __name__ == "__main__":
    import os, pandas as pd
    BASE = os.path.join(os.path.dirname(__file__), "..", "data")
    restaurants_df = pd.read_csv(os.path.join(BASE, "restaurants.csv"))

    features, Q = restaurant_feature_basis(restaurants_df)
    print(f"Input shape:          {features.shape}")
    print(f"Orthonormal basis Q:  {Q.shape}  (should be <= {features.shape[0]} rows)")
    print(f"Orthonormality check: {verify_orthonormality(Q)}")
    print(f"\nFirst basis vector:   {Q[0].round(4)}")
    print(f"Second basis vector:  {Q[1].round(4)}")

    # Sanity check: dot product of first two basis vectors should be ~0
    dot = np.dot(Q[0], Q[1])
    print(f"<e1, e2> = {dot:.2e}  (should be ~0)")
