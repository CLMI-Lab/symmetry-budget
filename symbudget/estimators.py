"""
Algorithm 2: constrained least squares on ``V_k`` via sufficient statistics.

The estimator is

    W_hat_k = argmin_{W in V_k} sum_i || y_i - W x_i ||^2 .

Writing ``W = sum_a theta_a B_a`` in the orthonormal orbit basis of
``V_k`` (Algorithm 1), the normal equations are

    G theta = r,    G_{ba} = <B_b, B_a Sigma_hat>,    r_b = <B_b, C>,

with ``Sigma_hat = X^T X`` and ``C = Y^T X``.  Both are ``d x d``, so the final
solve is independent of ``n``; forming the statistics is ``O(n d^2)`` and the
solve is ``O(p_k^3 + p_k d^3)``.  Per-architecture cost ``O(n d^2 + d^3)``
(Appendix F's "computational cost" note).

A relative ridge of ``1e-8 * tr(G) / p_k`` is added for numerical stability, as
in Appendix B; it changes no reported number at the quoted precision (verified
in ``tests/test_estimators.py``).

Configurations with ``n d < p_k + 2`` are under-determined and return NaN.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .lattice import SubgroupLattice, lattice

__all__ = ["FitResult", "fit_constrained", "excess_risk", "fit_all_k", "CheckpointStore"]

RIDGE_REL = 1e-8


@dataclass
class FitResult:
    k: int
    n: int
    p: int
    W: np.ndarray | None
    excess_risk: float
    underdetermined: bool = False

    def to_record(self) -> dict:
        r = asdict(self)
        r.pop("W")
        return r


def _gram(lat: SubgroupLattice, k: int, Sigma: np.ndarray) -> np.ndarray:
    """``G_{ba} = <B_b, B_a Sigma>`` for the orthonormal orbit basis of ``V_k``."""
    B = lat.basis(k)                       # (p, d, d)
    BS = B @ Sigma                         # (p, d, d) -- B_a Sigma
    p = B.shape[0]
    return B.reshape(p, -1) @ BS.reshape(p, -1).T


def fit_constrained(Sigma: np.ndarray, C: np.ndarray, n: int, k: int,
                    lat: SubgroupLattice, ridge_rel: float = RIDGE_REL) -> np.ndarray | None:
    """Solve the constrained least-squares problem; ``None`` if under-determined."""
    d = lat.d
    p = lat.p[k]
    if n * d < p + 2:
        return None
    G = _gram(lat, k, Sigma)
    r = lat.basis_matrix(k) @ C.ravel()
    if ridge_rel:
        G = G + np.eye(p) * (ridge_rel * np.trace(G) / p)
    theta = np.linalg.solve(G, r)
    return (theta[:, None, None] * lat.basis(k)).sum(axis=0)


def excess_risk(W_hat: np.ndarray | None, W_star: np.ndarray) -> float:
    """
    Population excess risk.  Because ``E[x x^T] = I``, the population risk gap of
    any ``W`` is exactly ``|| W - W* ||_F^2`` — no test set is needed (Section 2).
    """
    if W_hat is None:
        return float("nan")
    return float(np.sum((W_hat - W_star) ** 2))


def fit_all_k(Sigma: np.ndarray, C: np.ndarray, n: int, W_star: np.ndarray,
              lat: SubgroupLattice, keep_weights: bool = False) -> dict[int, FitResult]:
    """Fit every architecture in the lattice on one dataset."""
    out = {}
    for k in lat.ks:
        W = fit_constrained(Sigma, C, n, k, lat)
        out[k] = FitResult(
            k=k, n=n, p=lat.p[k],
            W=W if keep_weights else None,
            excess_risk=excess_risk(W, W_star),
            underdetermined=W is None,
        )
    return out


class CheckpointStore:
    """
    Checkpoint directory for fitted weights and stage-level caches.

    Two kinds of artefact are stored:

    * **model checkpoints** -- ``W_hat_k`` for a representative subset of
      configurations, as ``.npz`` with a JSON-serialisable metadata block, so a
      reader can reload a trained architecture and re-score it without re-running
      anything.  ``load_model`` returns ``(W_hat, meta)``.
    * **stage caches** -- the full result table of a finished experiment stage,
      so ``run_all.py`` can resume after an interruption.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        (self.root / "stages").mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------- model files

    def model_path(self, tag: str) -> Path:
        return self.root / "models" / f"{tag}.npz"

    def save_model(self, tag: str, W_hat: np.ndarray, W_star: np.ndarray, **meta) -> Path:
        path = self.model_path(tag)
        np.savez_compressed(
            path,
            W_hat=W_hat,
            W_star=W_star,
            meta=np.array(json.dumps(meta)),
        )
        return path

    def load_model(self, tag: str) -> tuple[np.ndarray, np.ndarray, dict]:
        z = np.load(self.model_path(tag), allow_pickle=False)
        return z["W_hat"], z["W_star"], json.loads(str(z["meta"]))

    def list_models(self) -> list[str]:
        return sorted(p.stem for p in (self.root / "models").glob("*.npz"))

    # ---------------------------------------------------------- stage caches

    def stage_path(self, stage: str) -> Path:
        return self.root / "stages" / f"{stage}.npz"

    def has_stage(self, stage: str) -> bool:
        return self.stage_path(stage).exists()

    def save_stage(self, stage: str, **arrays) -> Path:
        path = self.stage_path(stage)
        np.savez_compressed(path, **arrays)
        return path

    def load_stage(self, stage: str) -> dict:
        z = np.load(self.stage_path(stage), allow_pickle=False)
        return {k: z[k] for k in z.files}
