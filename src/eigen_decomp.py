"""Eigendecomposition and diagonalization utilities."""

"""
eigen_decomp.py — Eigendecomposition & Diagonalization
=======================================================
Person 3 | TasteVector Project

Computes the covariance matrix C = F^T F of the restaurant feature matrix,
then finds eigenvalues and eigenvectors:

    C = P D P^{-1}          (diagonalization)

where D = diag(eigenvalues) and P = matrix of eigenvectors (columns).

The top eigenvectors capture the feature directions along which restaurants
vary most (analogous to PCA principal components).  These are used to
project user preference vectors into the most informative subspace.

Also validates the Cayley-Hamilton theorem as a correctness check:
    substituting eigenvalue λ into the characteristic polynomial p(λ) → 0
"""

import numpy as np


# ── Covariance matrix ────────────────────────────────────────────────────────

def covariance_matrix(F: np.ndarray) -> np.ndarray:
    """
    Compute the feature covariance matrix C = F^T @ F.

    F : (n_restaurants, n_features)
    C : (n_features, n_features)  — symmetric, positive semi-definite
    """
    return F.T @ F


# ── Eigendecomposition ───────────────────────────────────────────────────────

def eigen_decompose(C: np.ndarray) -> tuple:
    """
    Compute eigenvalues and eigenvectors of a square matrix C.

    C v = λ v

    Returns
    -------
    eigenvalues  : (n,)     — may be complex for non-symmetric matrices
                             (C = F^T F is always symmetric → always real)
    eigenvectors : (n, n)   — columns are eigenvectors
    """
    eigenvalues, eigenvectors = np.linalg.eig(C)

    # sort descending by magnitude so the most important directions come first
    order = np.argsort(np.abs(eigenvalues))[::-1]
    return eigenvalues[order], eigenvectors[:, order]


# ── Diagonalization  C = P D P^{-1} ─────────────────────────────────────────

def diagonalize(C: np.ndarray) -> tuple:
    """
    Diagonalize C = P D P^{-1}.

    Returns
    -------
    P    : (n, n)  matrix whose columns are eigenvectors
    D    : (n, n)  diagonal matrix of eigenvalues
    P_inv: (n, n)  inverse of P
    """
    eigenvalues, P = eigen_decompose(C)
    D = np.diag(eigenvalues)
    P_inv = np.linalg.inv(P)
    return P, D, P_inv


def verify_diagonalization(C: np.ndarray, P, D, P_inv,
                           tol: float = 1e-8) -> bool:
    """
    Check  P @ D @ P^{-1}  ≈  C  within floating-point tolerance.
    Returns True if the reconstruction is accurate.
    """
    C_reconstructed = P @ D @ P_inv
    return bool(np.allclose(C_reconstructed, C, atol=tol))


# ── Cayley-Hamilton theorem ──────────────────────────────────────────────────

def cayley_hamilton_check(C: np.ndarray, tol: float = 1e-6) -> dict:
    """
    Cayley-Hamilton: every matrix satisfies its own characteristic polynomial.

    The characteristic polynomial of C is  det(C - λI) = 0.
    We compute the coefficients via numpy and evaluate p(C) — if the theorem
    holds, the result should be the zero matrix (within floating-point error).

    For each eigenvalue λ_i we also verify p(λ_i) ≈ 0.

    Returns
    -------
    dict with keys:
        'matrix_check' : bool   — p(C) ≈ 0 (Frobenius norm < tol * n^2)
        'eigenvalue_residuals' : array of |p(λ_i)| for each eigenvalue
        'all_pass'     : bool
    """
    n = C.shape[0]
    coeffs = np.poly(C)           # characteristic polynomial coefficients

    # Evaluate p(C) using Horner's method
    pC = np.zeros_like(C, dtype=complex)
    for c in coeffs:
        pC = pC @ C + c * np.eye(n)

    matrix_frobenius = np.linalg.norm(pC)
    matrix_ok = bool(matrix_frobenius < tol * n * n)

    # Evaluate p(λ) for each eigenvalue
    eigenvalues, _ = np.linalg.eig(C)
    residuals = np.array([abs(np.polyval(coeffs, lam)) for lam in eigenvalues])
    eigenvalue_ok = bool(np.all(residuals < tol))

    return {
        "matrix_check": matrix_ok,
        "matrix_frobenius_norm": float(matrix_frobenius),
        "eigenvalue_residuals": residuals,
        "all_pass": matrix_ok and eigenvalue_ok,
    }


# ── Top eigenvectors (PCA-style projection) ──────────────────────────────────

def top_k_eigenvectors(C: np.ndarray, k: int) -> np.ndarray:
    """
    Return the k eigenvectors corresponding to the k largest eigenvalues.
    These span the directions of maximum variance in the feature space.

    Returns
    -------
    E : (n_features, k)  — columns are the top-k eigenvectors
    """
    eigenvalues, eigenvectors = eigen_decompose(C)
    return eigenvectors[:, :k]


def project_onto_top_k(v: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    Project a preference vector v onto the subspace spanned by E.

    v_proj = E @ E^T @ v

    Parameters
    ----------
    v : (n_features,)  — user preference vector
    E : (n_features, k) — top-k eigenvectors from top_k_eigenvectors()

    Returns
    -------
    v_proj : (n_features,)  — projection of v onto the top-k subspace
    """
    return E @ (E.T @ v)


# ── Analysis report ──────────────────────────────────────────────────────────

def eigen_report(F: np.ndarray, feature_names: list = None) -> None:
    """
    Print a human-readable eigendecomposition report for the feature matrix F.
    """
    C = covariance_matrix(F)
    eigenvalues, eigenvectors = eigen_decompose(C)
    P, D, P_inv = diagonalize(C)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(F.shape[1])]

    print("=" * 55)
    print("EIGENDECOMPOSITION REPORT")
    print("=" * 55)
    print(f"\nCovariance matrix C = F^T F  shape: {C.shape}")

    total = np.abs(eigenvalues).sum()
    print("\nEigenvalues (sorted descending):")
    for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
        pct = 100 * abs(lam) / total
        top_feat = feature_names[np.argmax(np.abs(vec))]
        print(f"  λ_{i+1} = {lam.real:8.3f}  ({pct:5.1f}%)  "
              f"dominant feature: {top_feat}")

    diag_ok = verify_diagonalization(C, P, D, P_inv)
    print(f"\nDiagonalization C = P D P^{{-1}} verified: {diag_ok}")

    ch = cayley_hamilton_check(C)
    print(f"Cayley-Hamilton check passed:          {ch['all_pass']}")
    print(f"  p(C) Frobenius norm: {ch['matrix_frobenius_norm']:.2e}")


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(base, "src"))
    from data_loader import load_all

    restaurants, _, _ = load_all(os.path.join(base, "data"))
    F = restaurants[["price", "spice", "distance_km", "veg_friendly"]].to_numpy(dtype=float)
    feature_names = ["price", "spice", "distance_km", "veg_friendly"]

    eigen_report(F, feature_names)

    C = covariance_matrix(F)
    E = top_k_eigenvectors(C, k=2)
    print(f"\nTop-2 eigenvectors shape: {E.shape}")
    v = np.array([3.0, 4.0, 2.0, 1.0])   # example user preference vector
    v_proj = project_onto_top_k(v, E)
    print(f"User pref vector:  {v}")
    print(f"Projected onto k=2 subspace: {np.round(v_proj, 4)}")
