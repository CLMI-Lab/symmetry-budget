"""
Experiment configuration.

Three presets:

``paper``  the protocol of Appendix B -- 40 seeds, d in {8, 12, 16}, full grids.
``quick``  same structure, 8 seeds and coarser grids; ~1 minute, for development.
``smoke``  minimal; used by the test-suite and CI.

Every stage reads its grid from here, so the mapping from a reported number to
the configuration that produced it is one-to-one (the paper's reproducibility
claim).  ``Config.fingerprint()`` hashes the whole configuration into the
results file so stale checkpoints can never be silently reused.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

import numpy as np

__all__ = ["Config", "PRESETS", "get_config"]


def _geom(lo, hi, m):
    return [float(x) for x in np.geomspace(lo, hi, m)]


@dataclass
class Config:
    name: str = "paper"

    # ---- common
    d: int = 12
    sigma: float = 0.5
    n_seeds: int = 40
    seed0: int = 0

    # ---- phase diagram (Sec. 4.1)
    phase_n: list[int] = field(default_factory=lambda: [20, 30, 45, 70, 100, 150, 250, 400, 700, 1200, 2000, 4000])
    phase_eps: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4, 0.8])

    # ---- crossover / exponent (Sec. 4.2)
    cross_eps: list[float] = field(default_factory=lambda: _geom(0.05, 0.8, 9))
    cross_n_lo: float = 15.0
    cross_n_hi: float = 30000.0
    cross_n_points: int = 60

    # ---- generic-defect table (Table 1)
    table1_eps: float = 0.8
    table1_seeds: int = 40

    # ---- dimension dependence (Sec. 4.3)
    dims: list[int] = field(default_factory=lambda: [8, 16])
    dim_n: dict = field(default_factory=lambda: {
        8: [20, 45, 100, 250, 700, 1200, 2000, 4000],
        16: [20, 30, 45, 70, 100, 150, 250, 400, 700, 1200, 2000, 4000],
    })
    dim_eps: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    dim_cross_seeds: int = 20
    dim_cross_n_points: int = 40

    # ---- graded defect (Sec. 4.4)
    graded_k0: int = 4
    graded_b: float = 0.35
    graded_c: float = 0.08
    graded_n: list[int] = field(default_factory=lambda: [
        20, 30, 50, 80, 120, 200, 350, 600, 1000, 2500, 5000, 10000, 20000, 40000])

    # ---- nonlinear control (Sec. 4.5)
    nl_sigma: float = 0.3
    nl_seeds: int = 10
    nl_steps: int = 3000
    nl_lr: float = 0.05
    nl_width: int = 12
    nl_test: int = 4000
    nl_n: list[int] = field(default_factory=lambda: [50, 100, 200, 400, 800, 1600, 2400, 3200])
    nl_ks: list[int] = field(default_factory=lambda: [1, 3, 12])
    nl_eps: list[float] = field(default_factory=lambda: [0.1, 0.4])

    # ---- variance constant / coupling (Appendix C, F)
    var_n: int = 4000
    var_eps: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.4])
    coupling_eps: float = 0.8
    coupling_n: list[int] = field(default_factory=lambda: [100, 200, 400, 800, 1600])
    coupling_seeds: int = 400

    # ---- Proposition 8 audit
    nonabelian_groups: list[str] = field(default_factory=lambda: ["C4", "C6", "V4", "S3", "D4", "S4"])

    @property
    def seeds(self) -> list[int]:
        return list(range(self.seed0, self.seed0 + self.n_seeds))

    def cross_n_grid(self, points: int | None = None) -> np.ndarray:
        pts = points or self.cross_n_points
        g = np.unique(np.round(np.geomspace(self.cross_n_lo, self.cross_n_hi, pts)).astype(int))
        return g

    def fingerprint(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fingerprint"] = self.fingerprint()
        return d


def _quick() -> Config:
    c = Config(name="quick", n_seeds=8)
    c.phase_n = [20, 45, 100, 250, 700, 2000]
    c.phase_eps = [0.05, 0.2, 0.8]
    c.cross_eps = _geom(0.1, 0.8, 5)
    c.cross_n_hi = 8000.0
    c.cross_n_points = 30
    c.table1_seeds = 20
    c.dims = [8, 16]
    c.dim_n = {8: [20, 100, 700, 2000], 16: [20, 100, 400, 2000]}
    c.dim_eps = [0.1, 0.4]
    c.dim_cross_seeds = 8
    c.dim_cross_n_points = 24
    c.graded_n = [20, 50, 120, 350, 1000, 2500, 10000, 40000]
    c.nl_seeds = 4
    c.nl_steps = 800
    c.nl_n = [50, 200, 800, 3200]
    c.coupling_seeds = 120
    c.coupling_n = [200, 800]
    return c


def _smoke() -> Config:
    c = _quick()
    c.name = "smoke"
    c.n_seeds = 3
    c.phase_n = [45, 250, 2000]
    c.phase_eps = [0.1, 0.8]
    c.cross_eps = _geom(0.2, 0.8, 3)
    c.cross_n_hi = 3000.0
    c.cross_n_points = 16
    c.table1_seeds = 6
    c.dims = [8]
    c.dim_n = {8: [45, 250, 2000]}
    c.dim_eps = [0.2, 0.8]
    c.dim_cross_seeds = 3
    c.dim_cross_n_points = 12
    c.graded_n = [50, 350, 2500, 20000]
    c.nl_seeds = 2
    c.nl_steps = 200
    c.nl_n = [50, 400]
    c.nl_ks = [1, 12]
    c.nl_eps = [0.4]
    c.coupling_seeds = 40
    c.coupling_n = [200]
    c.nonabelian_groups = ["C4", "S3"]
    return c


PRESETS = {"paper": Config, "quick": _quick, "smoke": _smoke}


def get_config(name: str = "paper") -> Config:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]()
