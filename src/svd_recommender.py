"""SVD-based collaborative filtering routines."""

"""
svd_recommender.py — SVD-Based Collaborative Filtering
=======================================================
Person 3 | TasteVector Project

Decomposes the mean-centered rating matrix R using SVD:
    R = U * Sigma * V^T

Truncated SVD (top-k singular values) reduces noise and fills in
predicted scores for unrated user-restaurant pairs.
"""

import numpy as np


# ── Mean-centering ──────────────────────────────────────────────────────────

def mean_center(R: np.ndarray) -> tuple:
    """
    Subtract each user's mean rating (rated entries only) from their row.

    Parameters
    ----------
    R : (n_users, n_restaurants)  0 = unrated

    Returns
    -------
    R_centered : mean-centered matrix
    user_means : per-user mean, shape (n_users,)
    """
    R_centered = R.astype(float).copy()
    user_means = np.zeros(R.shape[0])
    for i in range(R.shape[0]):
        mask = R[i] != 0
        if mask.any():
            user_means[i] = R[i, mask].mean()
            R_centered[i, mask] -= user_means[i]
    return R_centered, user_means


def restore_means(R_approx: np.ndarray, user_means: np.ndarray) -> np.ndarray:
    """Add per-user means back to an approximated matrix."""
    return R_approx + user_means[:, np.newaxis]


# ── SVD decomposition ───────────────────────────────────────────────────────

def decompose(R_centered: np.ndarray) -> tuple:
    """
    Full SVD:  R_centered = U @ diag(sigma) @ Vt

    Returns
    -------
    U     : (n_users, n_users)
    sigma : singular values in descending order, shape (min(m,n),)
    Vt    : (n_restaurants, n_restaurants)
    """
    U, sigma, Vt = np.linalg.svd(R_centered, full_matrices=True)
    return U, sigma, Vt


def truncate(U, sigma, Vt, k: int) -> tuple:
    """
    Keep only the top-k singular values/vectors (noise reduction).

    Returns
    -------
    U_k  : (n_users, k)
    s_k  : (k,)
    Vt_k : (k, n_restaurants)
    """
    k = min(k, len(sigma))
    return U[:, :k], sigma[:k], Vt[:k, :]


def reconstruct(U_k, s_k, Vt_k) -> np.ndarray:
    """R_approx = U_k @ diag(s_k) @ Vt_k"""
    return U_k @ np.diag(s_k) @ Vt_k


# ── Variance explained (helps choose k) ─────────────────────────────────────

def variance_explained(sigma: np.ndarray) -> np.ndarray:
    """Cumulative fraction of variance explained by each singular value."""
    sq = sigma ** 2
    return np.cumsum(sq) / sq.sum()


def recommended_k(sigma: np.ndarray, threshold: float = 0.80) -> int:
    """Smallest k that explains at least `threshold` of total variance."""
    cumvar = variance_explained(sigma)
    hits = np.where(cumvar >= threshold)[0]
    return int(hits[0]) + 1 if len(hits) > 0 else len(sigma)


# ── High-level prediction pipeline ──────────────────────────────────────────

def predict_ratings(R: np.ndarray, k: int = 4) -> np.ndarray:
    """
    Full pipeline: mean-center → SVD → truncate → reconstruct → restore means.

    Returns
    -------
    R_pred : (n_users, n_restaurants)  predicted score for every cell
    """
    R_centered, user_means = mean_center(R)
    U, sigma, Vt = decompose(R_centered)
    U_k, s_k, Vt_k = truncate(U, sigma, Vt, k)
    R_approx = reconstruct(U_k, s_k, Vt_k)
    return restore_means(R_approx, user_means)


def top_n_for_user(user_idx: int, R: np.ndarray, R_pred: np.ndarray,
                   n: int = 5, only_unrated: bool = True) -> list:
    """
    Top-N restaurant recommendations for a single user.

    Parameters
    ----------
    user_idx    : row index of the target user
    R           : original rating matrix (to identify already-rated items)
    R_pred      : predicted rating matrix
    only_unrated: exclude restaurants the user has already rated

    Returns
    -------
    List of (restaurant_idx, predicted_score) sorted descending by score
    """
    scores = R_pred[user_idx].copy()
    if only_unrated:
        scores[R[user_idx] != 0] = -np.inf
    top_idx = np.argsort(scores)[::-1][:n]
    return [(int(i), float(scores[i])) for i in top_idx]


# ── Evaluation ───────────────────────────────────────────────────────────────

def frobenius_error(R_original: np.ndarray, R_approx: np.ndarray) -> float:
    """
    Frobenius norm of residuals on rated entries only.
    Used in unit tests to verify SVD beats a random baseline.
    """
    mask = R_original != 0
    diff = (R_original - R_approx)[mask]
    return float(np.sqrt((diff ** 2).sum()))


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "src"))
    from data_loader import load_all
    from matrix_builder import build_rating_matrix

    restaurants, users, ratings = load_all(os.path.join(base, "data"))
    R = build_rating_matrix(users, restaurants, ratings)

    print(f"Rating matrix: {R.shape}  |  "
          f"Sparsity: {100*np.count_nonzero(R)/R.size:.1f}% rated")

    _, sigma, _ = decompose(mean_center(R)[0])
    print("\nSingular values:", np.round(sigma[:8], 3))

    k = recommended_k(sigma)
    print(f"Recommended k (80% variance): {k}")
    for i, v in enumerate(variance_explained(sigma)[:8]):
        print(f"  k={i+1}: {v*100:.1f}% variance")

    R_pred = predict_ratings(R, k=k)
    print("\nTop-5 recs for user 0 (Alice):")
    for rest_idx, score in top_n_for_user(0, R, R_pred, n=5):
        name = restaurants.iloc[rest_idx]["name"]
        print(f"  [{rest_idx+1}] {name:<25} predicted={score:.3f}")

    print(f"\nFrobenius error (rated entries): {frobenius_error(R, R_pred):.4f}")
