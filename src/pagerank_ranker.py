"""PageRank-style restaurant ranking utilities."""

"""
pagerank_ranker.py — PageRank-Style Restaurant Ranking
======================================================
Person 3 | TasteVector Project

Builds a restaurant-to-restaurant similarity graph where:
  - Each node  = a restaurant
  - Edge weight = cosine similarity between the two restaurants' feature vectors

The graph is represented as an (n_restaurants x n_restaurants) matrix G.

A PageRank-style score is derived from the dominant eigenvector of G
(the eigenvector corresponding to the largest eigenvalue).

Restaurants that are highly similar to many other well-regarded restaurants
receive a high PageRank score.  This global importance signal is combined
with the SVD collaborative-filtering score in recommender.py.
"""

import numpy as np


# ── Cosine similarity helpers ────────────────────────────────────────────────

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    sim(u, v) = (u · v) / (||u|| * ||v||)
    Returns 0.0 if either vector is the zero vector.
    """
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


# ── Similarity graph construction ────────────────────────────────────────────

def build_similarity_graph(F: np.ndarray) -> np.ndarray:
    """
    Build a restaurant-to-restaurant similarity graph G.

    G[i, j] = cosine_similarity(F[i], F[j])

    Parameters
    ----------
    F : (n_restaurants, n_features)

    Returns
    -------
    G : (n_restaurants, n_restaurants)  symmetric, values in [0, 1]
        diagonal entries set to 0 (a restaurant is not similar to itself
        for ranking purposes)
    """
    n = F.shape[0]
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(F[i], F[j])
            G[i, j] = sim
            G[j, i] = sim          # symmetric
    return G


def normalize_graph(G: np.ndarray) -> np.ndarray:
    """
    Row-normalize G so each row sums to 1 (stochastic matrix).
    Rows that are all-zero are left as zero (isolated nodes).
    This is analogous to the PageRank transition matrix.
    """
    row_sums = G.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # avoid division by zero
    return G / row_sums


# ── PageRank via dominant eigenvector ────────────────────────────────────────

def dominant_eigenvector(M: np.ndarray) -> np.ndarray:
    """
    Find the dominant eigenvector of matrix M using numpy.linalg.eig.

    The dominant eigenvector corresponds to the eigenvalue with the
    largest absolute value.  For a non-negative stochastic matrix this
    is the Perron-Frobenius eigenvector (all non-negative entries).

    Returns
    -------
    v : (n,)  real-valued dominant eigenvector, L1-normalized so scores
              sum to 1 and are interpretable as probabilities.
    """
    eigenvalues, eigenvectors = np.linalg.eig(M)
    idx = np.argmax(np.abs(eigenvalues))
    v = eigenvectors[:, idx].real          # take real part (tiny imaginary residuals)
    v = np.abs(v)                          # Perron-Frobenius: entries should be ≥ 0
    v /= v.sum() if v.sum() != 0 else 1   # L1-normalize → interpretable scores
    return v


def pagerank_scores(F: np.ndarray) -> np.ndarray:
    """
    Full pipeline: build graph → normalize → dominant eigenvector.

    Parameters
    ----------
    F : (n_restaurants, n_features)

    Returns
    -------
    scores : (n_restaurants,)  PageRank importance score per restaurant,
             normalized to sum to 1.
    """
    G = build_similarity_graph(F)
    G_norm = normalize_graph(G)
    return dominant_eigenvector(G_norm)


# ── Power iteration (alternative / validation) ───────────────────────────────

def power_iteration(M: np.ndarray, n_iter: int = 100,
                    tol: float = 1e-10) -> np.ndarray:
    """
    Compute the dominant eigenvector via power iteration.
    More numerically stable for large matrices; useful for validating
    the numpy.linalg.eig result.

    Starts from a uniform vector and repeatedly multiplies by M^T,
    converging to the stationary distribution of the Markov chain.
    """
    n = M.shape[0]
    v = np.ones(n) / n              # uniform start
    for _ in range(n_iter):
        v_new = M.T @ v
        norm = np.linalg.norm(v_new)
        if norm == 0:
            break
        v_new /= norm
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    v = np.abs(v)
    v /= v.sum() if v.sum() != 0 else 1
    return v


# ── Ranked restaurant list ────────────────────────────────────────────────────

def rank_restaurants(scores: np.ndarray) -> list:
    """
    Return restaurant indices sorted by PageRank score (descending).

    Returns
    -------
    List of (restaurant_idx, score) tuples
    """
    order = np.argsort(scores)[::-1]
    return [(int(i), float(scores[i])) for i in order]


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "src"))
    from data_loader import load_all

    restaurants, _, _ = load_all(os.path.join(base, "data"))
    F = restaurants[["price", "spice", "distance_km", "veg_friendly"]].to_numpy(dtype=float)

    G = build_similarity_graph(F)
    print(f"Similarity graph shape: {G.shape}")
    print(f"Mean edge weight: {G[G > 0].mean():.4f}")

    scores_eig  = pagerank_scores(F)
    G_norm = normalize_graph(G)
    scores_pow  = power_iteration(G_norm)

    print(f"\nCorrelation (eig vs power iter): "
          f"{np.corrcoef(scores_eig, scores_pow)[0,1]:.6f}")

    print("\nPageRank Restaurant Ranking:")
    print(f"{'Rank':<6}{'Name':<25}{'Score':>8}")
    print("-" * 40)
    for rank, (idx, score) in enumerate(rank_restaurants(scores_eig), 1):
        name = restaurants.iloc[idx]["name"]
        print(f"{rank:<6}{name:<25}{score:>8.5f}")
