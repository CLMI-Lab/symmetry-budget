#!/usr/bin/env python3
"""
Write ``results/csv/table6_provenance.csv`` -- the paper's Table 6.

Which reported quantities are evaluated in closed form (exact up to floating
point, independent of the seed list) and which are sample statistics.  Run after
``run_all.py``:

    python scripts/make_provenance.py --out results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROWS = [
    # quantity,                        origin,                              seed-dependent, module
    ("delta_k^2 (generic)", "eq. (4), closed form", False,
     "theory.delta2_generic"),
    ("delta_k^2 (graded, eq. 7)", "eq. (7), closed form", False,
     "theory.delta2_graded"),
    ("delta_k^2 (graded, exact lattice-wide)", "dimension count, closed form", False,
     "theory.delta2_graded_exact"),
    ("p_k = d^2 / k", "Proposition 1, closed form", False, "lattice.SubgroupLattice.p"),
    ("Pi_k", "Proposition 1, closed form", False, "lattice.SubgroupLattice.project"),
    ("n* (prediction)", "eq. (6), closed form", False, "theory.n_star"),
    ("R(u) (leading order)", "eq. (5), closed form", False, "theory.risk_leading_order"),
    ("R_fs(u) (finite sample)", "eq. (3), closed form given delta^2", False,
     "theory.risk_finite_sample"),
    ("graded window (8)", "Proposition 7, closed form", False, "theory.graded_window"),
    ("p_H, gamma_H (non-abelian)", "character/trace computation, closed form", False,
     "nonabelian.commutant_dim_trace"),
    ("Measured risk ||W_hat - W*||_F^2", "sample statistic", True,
     "estimators.excess_risk"),
    ("Measured k*", "argmin over the grid of a sample statistic", True,
     "experiments._phase_diagram"),
    ("Measured n* (median over seeds)", "sample statistic on a discrete log grid", True,
     "experiments._crossover_for_d"),
    ("Fitted exponent and its standard error", "OLS on sample statistics", True,
     "experiments._fit_exponent"),
    ("gamma_k (measured)", "sample statistic over resampled defect directions", True,
     "experiments.run_generic_defect"),
    ("rho_k", "sample statistic", True, "experiments.run_variance_constant"),
    ("c_k (coupling coefficient)", "weighted fit to sample statistics", True,
     "experiments.run_variance_constant"),
    ("Nonlinear test MSE", "sample statistic", True, "nonlinear.train_projected"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out) / "csv"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ROWS, columns=["quantity", "origin", "depends_on_seeds", "implemented_in"])
    path = out / "table6_provenance.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path} ({len(df)} rows; "
          f"{int((~df.depends_on_seeds).sum())} closed-form, "
          f"{int(df.depends_on_seeds.sum())} sampled)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
