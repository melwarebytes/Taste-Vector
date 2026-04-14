"""
similarity.py  —  Person 2: Similarity & Vector Space Engine
Computes cosine similarity between users and items using inner products and norms.

Linear Algebra concepts:
  - Inner product (dot product): <u, v> = u^T v
  - L2 norm: ||v|| = sqrt(<v, v>)
  - Cosine similarity: cos(u, v) = <u, v> / (||u|| * ||v||)
  - User-user and item-item similarity matrices
"""

import numpy as np
import pandas as pd


def build_rating_matrix(ratings_df: pd.DataFrame,
                        users_df: pd.DataFrame,
                        restaurants_df: pd.DataFrame) -> tuple[np.ndarray, list, list]:
    """
    Construct the user-restaurant rating matrix R (users x restaurants).
    Missing ratings are filled with 0.

    Returns:
        R        : ndarray of shape (n_users, n_restaurants)
        user_ids : ordered list of user IDs (row index)
        rest_ids : ordered list of restaurant IDs (column index)
    """
    user_ids = users_df["user_id"].tolist()
    rest_ids = restaurants_df["restaurant_id"].tolist()

    user_idx = {uid: i for i, uid in enumerate(user_ids)}
    rest_idx = {rid: j for j, rid in enumerate(rest_ids)}

    R = np.zeros((len(user_ids), len(rest_ids)))

    for _, row in ratings_df.iterrows():
        i = user_idx.get(row["user_id"])
        j = rest_idx.get(row["restaurant_id"])
        if i is not None and j is not None:
            R[i, j] = float(row["rating"])

    return R, user_ids, rest_ids


def cosine_similarity_vectors(u: np.ndarray, v: np.ndarray) -> float:
    """
    Cosine similarity between two 1-D vectors.

    cos(u, v) = <u, v> / (||u||_2 * ||v||_2)

    Returns 0.0 if either vector is the zero vector (undefined angle).
    """
    dot = np.dot(u, v)          # inner product
    norm_u = np.linalg.norm(u)  # L2 norm of u
    norm_v = np.linalg.norm(v)  # L2 norm of v

    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(dot / (norm_u * norm_v))


def user_user_similarity(R: np.ndarray) -> np.ndarray:
    """
    Compute the n_users x n_users cosine similarity matrix.
    Row i of R is the rating vector for user i.

    cos_sim[i, j] = <R[i], R[j]> / (||R[i]|| * ||R[j]||)
    """
    n_users = R.shape[0]
    sim = np.zeros((n_users, n_users))

    for i in range(n_users):
        for j in range(i, n_users):
            s = cosine_similarity_vectors(R[i], R[j])
            sim[i, j] = s
            sim[j, i] = s  # symmetric

    return sim


def item_item_similarity(R: np.ndarray) -> np.ndarray:
    """
    Compute the n_restaurants x n_restaurants cosine similarity matrix.
    Column j of R is the rating vector for restaurant j.

    cos_sim[j, k] = <R[:, j], R[:, k]> / (||R[:, j]|| * ||R[:, k]||)
    """
    n_items = R.shape[1]
    sim = np.zeros((n_items, n_items))

    for j in range(n_items):
        for k in range(j, n_items):
            s = cosine_similarity_vectors(R[:, j], R[:, k])
            sim[j, k] = s
            sim[k, j] = s

    return sim


def top_k_similar_users(user_sim: np.ndarray, user_idx: int, k: int = 5) -> list[tuple[int, float]]:
    """Return top-k most similar users to user_idx (excluding themselves)."""
    scores = [(j, user_sim[user_idx, j]) for j in range(user_sim.shape[0]) if j != user_idx]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def top_k_similar_items(item_sim: np.ndarray, item_idx: int, k: int = 5) -> list[tuple[int, float]]:
    """Return top-k most similar restaurants to item_idx (excluding itself)."""
    scores = [(j, item_sim[item_idx, j]) for j in range(item_sim.shape[0]) if j != item_idx]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


if __name__ == "__main__":
    import os
    BASE = os.path.join(os.path.dirname(__file__), "..", "data")

    ratings_df     = pd.read_csv(os.path.join(BASE, "ratings.csv"))
    users_df       = pd.read_csv(os.path.join(BASE, "users.csv"))
    restaurants_df = pd.read_csv(os.path.join(BASE, "restaurants.csv"))

    R, user_ids, rest_ids = build_rating_matrix(ratings_df, users_df, restaurants_df)
    print(f"Rating matrix shape: {R.shape}  (users x restaurants)")

    user_sim = user_user_similarity(R)
    item_sim = item_item_similarity(R)

    print("\nTop 3 users similar to U01:")
    for idx, score in top_k_similar_users(user_sim, 0, k=3):
        print(f"  {user_ids[idx]}  cos_sim={score:.4f}")

    print("\nTop 3 restaurants similar to R01:")
    for idx, score in top_k_similar_items(item_sim, 0, k=3):
        print(f"  {rest_ids[idx]}  cos_sim={score:.4f}")
