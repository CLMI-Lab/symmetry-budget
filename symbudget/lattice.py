"""
Subgroup lattice of C_d and the commutant spaces V_k  (Section 2 of the paper).

Implements
----------
Algorithm 1 (orbit basis)
    For k | d let s = d/k and let T = S^s act diagonally on index pairs by
    (i, j) -> (i + s, j + s) mod d.  V_k = {W : W T = T W} is exactly the space
    of matrices constant along the orbits of that action.  The action is free,
    so every orbit has k elements and there are d^2 / k of them; the normalised
    orbit indicators form an orthonormal basis of V_k (Proposition 1).

Proposition 1  dim V_k = d^2 / k  and  Pi_k(W) = (1/k) sum_m S^{-m d/k} W S^{m d/k}.
Proposition 2  C_h <= C_k  =>  V_k subset V_h  =>  delta_h <= delta_k.

Everything here is exact linear algebra; nothing is estimated.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

__all__ = [
    "divisors",
    "shift_matrix",
    "orbit_labels",
    "SubgroupLattice",
]


def divisors(d: int) -> list[int]:
    """Ascending list of positive divisors of ``d``: the subgroup lattice index set."""
    if d < 1:
        raise ValueError("d must be a positive integer")
    return [k for k in range(1, d + 1) if d % k == 0]


def shift_matrix(d: int, power: int = 1) -> np.ndarray:
    """Permutation matrix of the cyclic shift ``(S x)_i = x_{i-1 mod d}``, raised to ``power``."""
    S = np.zeros((d, d))
    idx = np.arange(d)
    S[idx, (idx - 1) % d] = 1.0
    return np.linalg.matrix_power(S, power % d) if power % d else np.eye(d)


def orbit_labels(d: int, k: int) -> np.ndarray:
    """
    Label every index pair ``(i, j)`` by its orbit under ``(i, j) -> (i + s, j + s)``,
    ``s = d / k``.  Returns an integer array of shape ``(d, d)`` with labels
    ``0 .. d^2/k - 1``.  This is the combinatorial content of Proposition 1.
    """
    if d % k:
        raise ValueError(f"{k} does not divide {d}")
    s = d // k
    labels = -np.ones((d, d), dtype=np.int64)
    nxt = 0
    for i in range(d):
        for j in range(d):
            if labels[i, j] >= 0:
                continue
            a, b = i, j
            for _ in range(k):
                labels[a, b] = nxt
                a, b = (a + s) % d, (b + s) % d
            nxt += 1
    assert nxt == d * d // k, "orbit count must equal d^2/k (Proposition 1)"
    return labels


class SubgroupLattice:
    """
    The family of architectures ``{V_k : k | d}`` with orthonormal orbit bases.

    Attributes
    ----------
    d : int
    ks : list[int]                 divisors of d, ascending (k = 1 dense ... k = d convolutional)
    p : dict[int, int]             parameter counts p_k = d^2 / k
    """

    def __init__(self, d: int):
        self.d = int(d)
        self.ks = divisors(self.d)
        self.p = {k: self.d * self.d // k for k in self.ks}
        self._basis: dict[int, np.ndarray] = {}
        self._labels: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------ bases

    def labels(self, k: int) -> np.ndarray:
        if k not in self._labels:
            self._labels[k] = orbit_labels(self.d, k)
        return self._labels[k]

    def basis(self, k: int) -> np.ndarray:
        """
        Orthonormal basis of ``V_k`` as an array of shape ``(p_k, d, d)``.

        ``B_a = k^{-1/2} * 1[(i, j) in orbit a]`` — orthonormal because the orbits
        partition the index set and each has exactly ``k`` elements.
        """
        if k not in self._basis:
            lab = self.labels(k)
            p = self.p[k]
            B = np.zeros((p, self.d, self.d))
            B[lab.ravel(), np.repeat(np.arange(self.d), self.d), np.tile(np.arange(self.d), self.d)] = 1.0
            B /= np.sqrt(k)
            self._basis[k] = B
        return self._basis[k]

    def basis_matrix(self, k: int) -> np.ndarray:
        """The basis flattened to ``(p_k, d^2)`` (row-major ``vec``)."""
        return self.basis(k).reshape(self.p[k], self.d * self.d)

    # ------------------------------------------------------------ projections

    def project(self, W: np.ndarray, k: int) -> np.ndarray:
        """
        Orthogonal projection ``Pi_k(W)`` onto ``V_k``.

        Computed as the orbit average, which is the closed form of Proposition 1;
        :func:`project_by_group_average` gives the equivalent group-average form and
        the two are checked against each other in the test-suite.
        """
        lab = self.labels(k)
        sums = np.bincount(lab.ravel(), weights=np.asarray(W, dtype=float).ravel(), minlength=self.p[k])
        return (sums / k)[lab]

    def project_by_group_average(self, W: np.ndarray, k: int) -> np.ndarray:
        """``Pi_k(W) = (1/k) sum_{m<k} S^{-m d/k} W S^{m d/k}`` — the literal Proposition 1 formula."""
        s = self.d // k
        out = np.zeros_like(np.asarray(W, dtype=float))
        for m in range(k):
            T = shift_matrix(self.d, m * s)
            out += T.T @ W @ T
        return out / k

    def residual(self, W: np.ndarray, k: int) -> np.ndarray:
        """``(I - Pi_k) W``: the part of ``W`` that architecture ``k`` cannot represent."""
        return np.asarray(W, dtype=float) - self.project(W, k)

    def defect(self, W_star: np.ndarray, k: int) -> float:
        """``delta_k = || (I - Pi_k) W_star ||_F`` (Proposition 2)."""
        return float(np.linalg.norm(self.residual(W_star, k)))

    def defect_profile(self, W_star: np.ndarray) -> dict[int, float]:
        """``{k: delta_k^2}`` over the whole lattice."""
        return {k: self.defect(W_star, k) ** 2 for k in self.ks}

    # ------------------------------------------------------------- utilities

    def is_equivariant(self, W: np.ndarray, k: int, tol: float = 1e-10) -> bool:
        T = shift_matrix(self.d, self.d // k)
        return bool(np.linalg.norm(W @ T - T @ W) < tol)

    def random_in(self, k: int, rng: np.random.Generator, normalise: bool = True) -> np.ndarray:
        """Random element of ``V_k`` (unit Frobenius norm by default)."""
        W = self.project(rng.standard_normal((self.d, self.d)), k)
        if normalise:
            W /= np.linalg.norm(W)
        return W

    def random_in_complement(
        self, k: int, rng: np.random.Generator, inside: int | None = None, normalise: bool = True
    ) -> np.ndarray:
        """
        Random element of ``V_k^perp`` — optionally intersected with ``V_inside``,
        i.e. a draw from ``V_inside ∩ V_k^perp`` (used to build graded targets).
        """
        W = rng.standard_normal((self.d, self.d))
        if inside is not None:
            W = self.project(W, inside)
        W = self.residual(W, k)
        if normalise:
            W /= np.linalg.norm(W)
        return W

    def subgroup_order(self, k: int) -> int:
        return k

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SubgroupLattice(d={self.d}, ks={self.ks}, p={[self.p[k] for k in self.ks]})"


@lru_cache(maxsize=16)
def lattice(d: int) -> SubgroupLattice:
    """Cached lattice constructor (bases are expensive enough to be worth reusing)."""
    return SubgroupLattice(d)
