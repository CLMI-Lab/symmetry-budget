"""
The data model of eq. (2):

    x ~ N(0, I_d),   y = W* x + sigma xi,   xi ~ N(0, I_d),
    W* = W_eq + eps R,   W_eq in V_d,  R in V_d^perp,  ||W_eq||_F = ||R||_F = 1.

Two target families are provided:

* :func:`generic_target`  -- ``R`` uniform on the unit sphere of ``V_d^perp``
  (Proposition 4).
* :func:`graded_target`   -- ``W* = u_d + b u_{k0} + c u_1`` (Proposition 7).

The equivariant component is fixed across all experiments as a random unit-norm
circulant filter drawn from seed 0, exactly as Section 2 of the paper specifies.

Sufficient statistics
---------------------
Constrained least squares depends on the data only through
``Sigma_hat = X^T X`` (d x d) and ``C = Y^T X`` (d x d).  :class:`StreamingStats`
accumulates these in blocks, which is what makes the nested-``n`` grids of the
crossover experiment cheap: one pass over the largest sample size yields the
statistics at every grid point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import SubgroupLattice, lattice

__all__ = [
    "equivariant_component",
    "generic_target",
    "graded_target",
    "sample",
    "StreamingStats",
    "Target",
]


@dataclass
class Target:
    """A ground-truth map together with the metadata needed to reproduce it."""

    W: np.ndarray
    d: int
    kind: str
    eps: float = 0.0
    k0: int | None = None
    b: float = 0.0
    c: float = 0.0
    seed: int | None = None

    def defect_profile(self) -> dict[int, float]:
        lat = lattice(self.d)
        return lat.defect_profile(self.W)


def equivariant_component(d: int, seed: int = 0) -> np.ndarray:
    """
    The fixed unit-norm circulant filter ``W_eq in V_d`` (seed 0 by default, per the paper).

    A circulant matrix is built from a random filter and normalised, so it is
    exactly ``C_d``-equivariant by construction rather than by projection.
    """
    rng = np.random.default_rng(seed)
    h = rng.standard_normal(d)
    idx = np.arange(d)
    W = h[(idx[:, None] - idx[None, :]) % d]
    return W / np.linalg.norm(W)


def generic_target(d: int, eps: float, seed: int, lat: SubgroupLattice | None = None) -> Target:
    """``W* = W_eq + eps R`` with ``R`` uniform on the unit sphere of ``V_d^perp``."""
    lat = lat or lattice(d)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((d, d))
    R = lat.residual(R, d)                       # push into V_d^perp
    R /= np.linalg.norm(R)
    W = equivariant_component(d) + eps * R
    return Target(W=W, d=d, kind="generic", eps=eps, seed=seed)


def graded_target(d: int, k0: int, b: float, c: float, seed: int,
                  lat: SubgroupLattice | None = None) -> Target:
    """
    ``W* = u_d + b u_{k0} + c u_1`` with ``u_d in V_d``, ``u_{k0} in V_{k0} ∩ V_d^perp``,
    ``u_1 in V_{k0}^perp``, all unit norm and mutually orthogonal (Proposition 7).
    """
    lat = lat or lattice(d)
    rng = np.random.default_rng(seed)

    u_d = equivariant_component(d)
    u_k0 = lat.random_in_complement(d, rng, inside=k0)     # in V_{k0}, orthogonal to V_d
    u_1 = lat.random_in_complement(k0, rng)                # orthogonal to V_{k0}

    W = u_d + b * u_k0 + c * u_1
    return Target(W=W, d=d, kind="graded", k0=k0, b=b, c=c, seed=seed)


def sample(W_star: np.ndarray, n: int, sigma: float, rng: np.random.Generator):
    """Draw ``n`` i.i.d. pairs from eq. (2).  Returns ``(X, Y)`` of shape ``(n, d)``."""
    d = W_star.shape[0]
    X = rng.standard_normal((n, d))
    Y = X @ W_star.T + sigma * rng.standard_normal((n, d))
    return X, Y


class StreamingStats:
    """
    Block accumulator for the sufficient statistics ``Sigma_hat = X^T X`` and ``C = Y^T X``.

    Lets a nested grid ``n_1 < n_2 < ... < n_m`` be evaluated in a single pass:
    ``advance_to(n_j)`` draws only the ``n_j - n_{j-1}`` new samples.
    """

    def __init__(self, W_star: np.ndarray, sigma: float, rng: np.random.Generator,
                 block: int = 20000):
        self.W_star = np.asarray(W_star, dtype=float)
        self.d = self.W_star.shape[0]
        self.sigma = float(sigma)
        self.rng = rng
        self.block = int(block)
        self.n = 0
        self.Sigma = np.zeros((self.d, self.d))
        self.C = np.zeros((self.d, self.d))

    def advance_to(self, n: int) -> "StreamingStats":
        """Accumulate until the running sample count reaches ``n``."""
        if n < self.n:
            raise ValueError("StreamingStats only moves forward; restart for a smaller n")
        while self.n < n:
            m = min(self.block, n - self.n)
            X, Y = sample(self.W_star, m, self.sigma, self.rng)
            self.Sigma += X.T @ X
            self.C += Y.T @ X
            self.n += m
        return self

    @property
    def stats(self) -> tuple[np.ndarray, np.ndarray, int]:
        return self.Sigma, self.C, self.n
