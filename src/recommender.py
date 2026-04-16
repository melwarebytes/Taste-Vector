"""Main TasteVector recommendation pipeline."""

"""
recommender.py — Main Recommendation Pipeline
=============================================
Person 3 | TasteVector Project

Orchestrates the full recommendation pipeline:

  Existing user  →  SVD collaborative filtering + PageRank blending
  New user       →  Least Squares projection (cold-start) + PageRank blending

The final score is a weighted combination:
    final_score = α * svd_score + β * pagerank_score + γ * projection_score

Filters are applied post-scoring to remove restaurants that exceed the
user's max_price or max_distance constraints.
"""

import numpy as np
import pandas as pd
import os, sys

# Allow running this file directly from the src/ directory
_src = os.path.dirname(os.path.abspath(__file__))
if _src not in sys.path:
    sys.path.insert(0, _src)

from svd_recommender  import predict_ratings, top_n_for_user, mean_center, decompose, recommended_k
from eigen_decomp     import covariance_matrix, top_k_eigenvectors, project_onto_top_k
from pagerank_ranker  import pagerank_scores, build_similarity_graph, normalize_graph


# ── Weights for score blending ───────────────────────────────────────────────
ALPHA = 0.60   # SVD collaborative filtering weight
BETA  = 0.25   # PageRank global importance weight
GAMMA = 0.15   # Cold-start projection weight (only for new users)


# ── Feature matrix builder (inline; avoids circular imports) ─────────────────

def _build_feature_matrix(restaurants: pd.DataFrame) -> np.ndarray:
    cols = ["price", "spice", "distance_km", "veg_friendly"]
    return restaurants[cols].to_numpy(dtype=float)


# ── Constraint filtering ─────────────────────────────────────────────────────

def apply_constraints(restaurants: pd.DataFrame, scores: np.ndarray,
                      max_price: int = None,
                      max_distance: float = None) -> np.ndarray:
    """
    Set scores to -inf for restaurants that violate hard constraints.
    This ensures they never appear in the top-N list.
    """
    filtered = scores.copy()
    if max_price is not None:
        mask = restaurants["price"].to_numpy() > max_price
        filtered[mask] = -np.inf
    if max_distance is not None:
        mask = restaurants["distance_km"].to_numpy() > max_distance
        filtered[mask] = -np.inf
    return filtered


# ── Existing-user recommendation ─────────────────────────────────────────────

def recommend_for_existing_user(
    user_idx: int,
    R: np.ndarray,
    restaurants: pd.DataFrame,
    n: int = 5,
    k: int = None,
    max_price: int = None,
    max_distance: float = None,
) -> list:
    """
    Recommendation pipeline for a user with rating history.

    Parameters
    ----------
    user_idx     : row index in R
    R            : (n_users, n_restaurants) raw rating matrix
    restaurants  : DataFrame with restaurant metadata
    n            : number of results to return
    k            : SVD latent dimensions (auto-selected if None)
    max_price    : hard constraint on price (1-5)
    max_distance : hard constraint on distance (km)

    Returns
    -------
    List of dicts, each with restaurant metadata + scores
    """
    F = _build_feature_matrix(restaurants)

    # ── SVD score ──────────────────────────────────────────────────────────
    R_centered, user_means = mean_center(R)
    _, sigma, _ = decompose(R_centered)
    if k is None:
        k = recommended_k(sigma)
    R_pred = predict_ratings(R, k=k)
    svd_scores = R_pred[user_idx]

    # Normalize SVD scores to [0, 1]
    svd_min, svd_max = svd_scores.min(), svd_scores.max()
    if svd_max > svd_min:
        svd_norm = (svd_scores - svd_min) / (svd_max - svd_min)
    else:
        svd_norm = np.zeros_like(svd_scores)

    # ── PageRank score ─────────────────────────────────────────────────────
    pr_scores = pagerank_scores(F)          # already normalized to sum=1

    # ── Blend ──────────────────────────────────────────────────────────────
    final = ALPHA * svd_norm + BETA * pr_scores

    # ── Apply constraints ──────────────────────────────────────────────────
    final = apply_constraints(restaurants, final, max_price, max_distance)

    # ── Exclude already-rated restaurants ──────────────────────────────────
    final[R[user_idx] != 0] = -np.inf

    # ── Collect top-N ──────────────────────────────────────────────────────
    top_idx = np.argsort(final)[::-1][:n]
    return _format_results(top_idx, final, svd_norm, pr_scores, restaurants)


# ── New-user recommendation (cold-start) ─────────────────────────────────────

def recommend_for_new_user(
    preferences: np.ndarray,
    R: np.ndarray,
    restaurants: pd.DataFrame,
    n: int = 5,
    max_price: int = None,
    max_distance: float = None,
) -> list:
    """
    Cold-start recommendation: user has no rating history.

    Strategy
    --------
    1. Project the user preference vector onto the top-k eigenvectors of
       the restaurant feature covariance matrix (Gram-Schmidt / Least Squares).
    2. Compute cosine similarity between the projected vector and each restaurant.
    3. Blend with PageRank global importance score.

    Parameters
    ----------
    preferences : (n_features,) user's stated preferences
                  [price, spice, distance_km, veg_friendly]
    """
    F = _build_feature_matrix(restaurants)
    n_rest = F.shape[0]

    # ── Eigendecomposition of feature covariance ───────────────────────────
    C = covariance_matrix(F)
    k = min(3, C.shape[0])
    E = top_k_eigenvectors(C, k=k)

    # ── Project user preference vector ─────────────────────────────────────
    v_proj = project_onto_top_k(preferences, E)

    # ── Cosine similarity of projected vector vs each restaurant ───────────
    proj_scores = np.array([
        _cosine(v_proj, F[j]) for j in range(n_rest)
    ])

    # ── PageRank score ─────────────────────────────────────────────────────
    pr_scores = pagerank_scores(F)

    # ── Blend ──────────────────────────────────────────────────────────────
    # For a new user there are no SVD scores, so GAMMA replaces ALPHA
    final = (ALPHA + GAMMA) * proj_scores + BETA * pr_scores

    # ── Apply constraints ──────────────────────────────────────────────────
    final = apply_constraints(restaurants, final, max_price, max_distance)

    top_idx = np.argsort(final)[::-1][:n]
    return _format_results(top_idx, final, proj_scores, pr_scores, restaurants)


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


# ── Result formatter ─────────────────────────────────────────────────────────

def _format_results(top_idx, final_scores, content_scores,
                    pr_scores, restaurants) -> list:
    results = []
    for rank, idx in enumerate(top_idx, 1):
        if final_scores[idx] == -np.inf:
            continue
        row = restaurants.iloc[idx]
        results.append({
            "rank":            rank,
            "restaurant_id":   int(row["restaurant_id"]),
            "name":            row["name"],
            "cuisine":         row["cuisine"],
            "price":           int(row["price"]),
            "spice":           int(row["spice"]),
            "distance_km":     float(row["distance_km"]),
            "veg_friendly":    bool(row["veg_friendly"]),
            "final_score":     round(float(final_scores[idx]), 4),
            "content_score":   round(float(content_scores[idx]), 4),
            "pagerank_score":  round(float(pr_scores[idx]), 4),
        })
    return results


# ── Public entry point (called by api.py) ────────────────────────────────────

def get_recommendations(
    user_id: int | None,
    preferences: dict,
    R: np.ndarray,
    users: pd.DataFrame,
    restaurants: pd.DataFrame,
    top_n: int = 5,
) -> list:
    """
    Main entry point.  Called by the Flask API.

    Parameters
    ----------
    user_id     : existing user ID, or None for a new/anonymous user
    preferences : dict with keys: max_price, spice_tolerance, max_distance
                  (cuisine is used for filtering in the API layer, not here)
    R           : rating matrix built by matrix_builder.py
    users       : users DataFrame
    restaurants : restaurants DataFrame
    top_n       : number of recommendations

    Returns
    -------
    List of recommendation dicts (see _format_results)
    """
    max_price    = preferences.get("max_price")
    max_distance = preferences.get("max_distance")
    spice        = preferences.get("spice_tolerance", 3)

    if user_id is not None:
        # Map user_id → row index
        uid_list = users["user_id"].tolist()
        if user_id in uid_list:
            user_idx = uid_list.index(user_id)
            return recommend_for_existing_user(
                user_idx, R, restaurants, n=top_n,
                max_price=max_price, max_distance=max_distance,
            )

    # New / anonymous user — use preference vector
    price_pref = max_price if max_price else 3
    dist_pref  = max_distance if max_distance else 3.0
    pref_vec   = np.array([price_pref, spice, dist_pref, 1.0], dtype=float)

    return recommend_for_new_user(
        pref_vec, R, restaurants, n=top_n,
        max_price=max_price, max_distance=max_distance,
    )


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "src"))
    from data_loader   import load_all
    from matrix_builder import build_rating_matrix

    restaurants, users, ratings = load_all(os.path.join(base, "data"))
    R = build_rating_matrix(users, restaurants, ratings)

    print("=" * 55)
    print("EXISTING USER — Alice (user_id=1)")
    print("=" * 55)
    recs = get_recommendations(
        user_id=1,
        preferences={"max_price": 4, "max_distance": 4.0, "spice_tolerance": 5},
        R=R, users=users, restaurants=restaurants, top_n=5,
    )
    for r in recs:
        print(f"  #{r['rank']} {r['name']:<25} "
              f"score={r['final_score']:.4f}  "
              f"(svd={r['content_score']:.4f}, pr={r['pagerank_score']:.4f})")

    print("\n" + "=" * 55)
    print("NEW USER — no history, prefers cheap + spicy + nearby")
    print("=" * 55)
    recs_new = get_recommendations(
        user_id=None,
        preferences={"max_price": 2, "max_distance": 2.0, "spice_tolerance": 5},
        R=R, users=users, restaurants=restaurants, top_n=5,
    )
    for r in recs_new:
        print(f"  #{r['rank']} {r['name']:<25} "
              f"score={r['final_score']:.4f}  "
              f"(proj={r['content_score']:.4f}, pr={r['pagerank_score']:.4f})")
