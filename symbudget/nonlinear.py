"""
Algorithm 4 (Section 4.5): the nonlinear control.

Student and teacher are both ``x -> W2 tanh(W1 x)``.  The student is trained by
full-batch gradient descent with *both* iterates projected onto ``V_k`` after
every step, so the network is exactly ``C_k``-equivariant throughout training
(not merely at convergence).  The teacher's weight matrices are drawn as in
eq. (2): an exactly equivariant part plus ``eps`` times a unit-norm defect in
``V_d^perp``.

This is a control, not evidence: a matched teacher, fixed-step GD, no tuning.
It can only show that the crossover is not destroyed by nonlinearity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import equivariant_component
from .lattice import SubgroupLattice, lattice

__all__ = ["TeacherTwoLayer", "make_teacher", "train_projected", "NonlinearRun"]


@dataclass
class TeacherTwoLayer:
    W1: np.ndarray
    W2: np.ndarray

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.tanh(X @ self.W1.T) @ self.W2.T


def _eq_plus_defect(d: int, eps: float, rng: np.random.Generator,
                    lat: SubgroupLattice, eq_seed: int) -> np.ndarray:
    R = lat.residual(rng.standard_normal((d, d)), d)
    R /= np.linalg.norm(R)
    return equivariant_component(d, seed=eq_seed) + eps * R


def make_teacher(d: int, eps: float, rng: np.random.Generator,
                 lat: SubgroupLattice | None = None) -> TeacherTwoLayer:
    """Teacher with both layers equal to ``W_eq + eps R`` (independent ``R`` per layer)."""
    lat = lat or lattice(d)
    return TeacherTwoLayer(
        W1=_eq_plus_defect(d, eps, rng, lat, eq_seed=0),
        W2=_eq_plus_defect(d, eps, rng, lat, eq_seed=1),
    )


@dataclass
class NonlinearRun:
    k: int
    n: int
    eps: float
    seed: int
    test_mse: float
    train_mse: float
    steps: int
    W1: np.ndarray | None = None
    W2: np.ndarray | None = None


def train_projected(teacher: TeacherTwoLayer, k: int, n: int, sigma: float, rng: np.random.Generator,
                    lat: SubgroupLattice, steps: int = 3000, lr: float = 0.05,
                    width: int | None = None, n_test: int = 4000,
                    keep_weights: bool = False, eps: float = float("nan"),
                    seed: int = -1) -> NonlinearRun:
    """
    Train a ``C_k``-equivariant two-layer tanh student against ``teacher``.

    Both weight matrices are initialised inside ``V_k`` and every gradient is
    projected onto ``V_k`` before the update, so the iterates never leave the
    constraint set.  Test MSE is the mean squared error per sample on ``n_test``
    fresh noiseless-input draws (noise is added to training targets only).
    """
    d = lat.d
    width = width or d

    X = rng.standard_normal((n, d))
    Y = teacher(X) + sigma * rng.standard_normal((n, d))
    Xte = rng.standard_normal((n_test, d))
    Yte = teacher(Xte)

    # initialise inside V_k
    W1 = lat.project(rng.standard_normal((d, d)) / np.sqrt(d), k)
    W2 = lat.project(rng.standard_normal((d, d)) / np.sqrt(d), k)

    for _ in range(steps):
        H = X @ W1.T
        A = np.tanh(H)
        P = A @ W2.T
        E = P - Y
        gP = (2.0 / n) * E
        gW2 = gP.T @ A
        gA = gP @ W2
        gH = gA * (1.0 - A * A)
        gW1 = gH.T @ X
        W2 -= lr * lat.project(gW2, k)
        W1 -= lr * lat.project(gW1, k)

    train_mse = float(np.mean(np.sum((np.tanh(X @ W1.T) @ W2.T - Y) ** 2, axis=1)) / d)
    test_mse = float(np.mean(np.sum((np.tanh(Xte @ W1.T) @ W2.T - Yte) ** 2, axis=1)) / d)

    return NonlinearRun(
        k=k, n=n, eps=eps, seed=seed, test_mse=test_mse, train_mse=train_mse, steps=steps,
        W1=W1 if keep_weights else None, W2=W2 if keep_weights else None,
    )
