"""
The experiment stages.  Each ``run_*`` function returns ``(tables, summary)``
where ``tables`` is a dict of pandas DataFrames written to ``results/csv`` and
``summary`` is a JSON-serialisable dict folded into ``results.json``.

Stage index
-----------
``theory_checks``     Propositions 1-2, projector identities, closed-form audits
``generic_defect``    Table 1 -- gamma_k closed form vs measurement
``phase_diagram``     Sec. 4.1 -- 60 cells at d = 12, predicted vs measured k*
``crossover``         Sec. 4.2 / Table 2 -- n*(eps), the eps^-2 exponent
``dimension``         Sec. 4.3 / Table 5 -- d in {8, 16}, the d(d-1) ratio
``graded``            Sec. 4.4 -- the 12 -> 4 -> 1 ladder and the window (8)
``nonlinear``         Sec. 4.5 / Table 3 -- two-layer tanh control
``variance_constant`` Appendices C, F -- rho_k, and a direct fit of the coupling
``nonabelian``        Appendix H -- Proposition 8 audit
"""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from . import theory as th
from .config import Config
from .data import StreamingStats, generic_target, graded_target, equivariant_component
from .estimators import CheckpointStore, fit_all_k, fit_constrained, excess_risk
from .lattice import lattice, divisors, shift_matrix
from .nonabelian import GROUPS, audit_group, interior_optimum_scan
from .nonlinear import make_teacher, train_projected

__all__ = [
    "run_theory_checks", "run_generic_defect", "run_phase_diagram", "run_crossover",
    "run_dimension", "run_graded", "run_nonlinear", "run_variance_constant",
    "run_nonabelian", "STAGES",
]


# --------------------------------------------------------------------- helpers


def _risk_curves(d: int, eps: float, seeds, n_grid, sigma: float, ks=None,
                 store: CheckpointStore | None = None, tag: str = "",
                 kind: str = "generic", **target_kw) -> np.ndarray:
    """
    Measured excess risk on a nested ``n`` grid.

    Returns an array of shape ``(len(seeds), len(n_grid), len(ks))``.  One
    :class:`StreamingStats` per seed walks the grid, so the cost is a single
    pass over ``max(n_grid)`` rather than one pass per grid point.

    The symmetry-breaking direction ``R`` is held **fixed** across seeds (it is
    drawn from a dedicated defect seed), and only the training samples are
    resampled -- the convention of Sections 4.1-4.2 and Appendix E.
    """
    lat = lattice(d)
    ks = ks or lat.ks
    if kind == "generic":
        target = generic_target(d, eps, seed=10_000 + int(round(eps * 1e6)), lat=lat)
    else:
        target = graded_target(d, target_kw["k0"], target_kw["b"], target_kw["c"], seed=7, lat=lat)
    W_star = target.W

    out = np.full((len(seeds), len(n_grid), len(ks)), np.nan)
    for si, seed in enumerate(seeds):
        ss = StreamingStats(W_star, sigma, np.random.default_rng(1_000_000 + seed))
        for ni, n in enumerate(n_grid):
            ss.advance_to(int(n))
            S, C, nn = ss.stats
            for kj, k in enumerate(ks):
                W = fit_constrained(S, C, nn, k, lat)
                out[si, ni, kj] = excess_risk(W, W_star)
                if store is not None and seed == seeds[0] and W is not None and tag:
                    if int(n) in (n_grid[0], n_grid[len(n_grid) // 2], n_grid[-1]):
                        store.save_model(
                            f"{tag}_d{d}_eps{eps:g}_n{int(n)}_k{k}", W, W_star,
                            d=d, k=k, n=int(n), eps=float(eps), sigma=sigma, seed=int(seed),
                            kind=kind, p=lat.p[k], excess_risk=float(out[si, ni, kj]),
                            delta2=float(lat.defect(W_star, k) ** 2),
                        )
    return out, target


def _first_crossing(risk_dense: np.ndarray, risk_equiv: np.ndarray, n_grid: np.ndarray) -> float:
    """First grid point at which the dense model's risk falls below the equivariant one's."""
    below = np.where(risk_dense < risk_equiv)[0]
    return float(n_grid[below[0]]) if len(below) else float("nan")


def _fit_exponent(eps: np.ndarray, n_star: np.ndarray) -> dict:
    """OLS of ``log n*`` on ``log eps`` with the standard error of the slope."""
    m = np.isfinite(n_star) & (n_star > 0)
    x, y = np.log(np.asarray(eps)[m]), np.log(np.asarray(n_star)[m])
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(cov[1, 1]))
    return {
        "exponent": float(coef[1]), "intercept": float(coef[0]), "stderr": se,
        "ci95": [float(coef[1] - 1.96 * se), float(coef[1] + 1.96 * se)],
        "n_points": int(len(x)),
    }


# ---------------------------------------------------------------- 1. theory


def run_theory_checks(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """Propositions 1 and 2 and the projector identities, verified exactly."""
    rows = []
    rng = np.random.default_rng(0)
    for d in sorted({cfg.d, *cfg.dims}):
        lat = lattice(d)
        W = rng.standard_normal((d, d))
        for k in lat.ks:
            B = lat.basis_matrix(k)
            P1 = lat.project(W, k)
            P2 = lat.project_by_group_average(W, k)
            rows.append({
                "d": d, "k": k,
                "p_k_formula": d * d // k,
                "p_k_rank": int(np.linalg.matrix_rank(B)),
                "basis_orthonormal_err": float(np.abs(B @ B.T - np.eye(B.shape[0])).max()),
                "orbit_vs_group_average_err": float(np.abs(P1 - P2).max()),
                "idempotent_err": float(np.abs(lat.project(P1, k) - P1).max()),
                "selfadjoint_err": float(abs(np.sum(lat.project(W, k) * W) - np.sum(P1 * P1))),
                "equivariance_err": float(np.linalg.norm(
                    P1 @ shift_matrix(d, d // k) - shift_matrix(d, d // k) @ P1)),
            })
    t_prop1 = pd.DataFrame(rows)

    # Proposition 2: monotonicity along the lattice (h | k  =>  delta_h <= delta_k)
    mono_rows = []
    for d in sorted({cfg.d, *cfg.dims}):
        lat = lattice(d)
        for trial in range(20):
            W_star = generic_target(d, 0.5, seed=trial, lat=lat).W
            prof = lat.defect_profile(W_star)
            for h in lat.ks:
                for k in lat.ks:
                    if k % h == 0:            # C_h <= C_k
                        mono_rows.append({
                            "d": d, "h": h, "k": k, "trial": trial,
                            "delta_h": np.sqrt(prof[h]), "delta_k": np.sqrt(prof[k]),
                            "ok": bool(np.sqrt(prof[h]) <= np.sqrt(prof[k]) + 1e-12),
                        })
    t_prop2 = pd.DataFrame(mono_rows)

    # endpoints: delta_1 = 0, delta_d = eps
    end_rows = []
    for d in sorted({cfg.d, *cfg.dims}):
        lat = lattice(d)
        for eps in [0.05, 0.2, 0.8]:
            W_star = generic_target(d, eps, seed=3, lat=lat).W
            end_rows.append({"d": d, "eps": eps,
                             "delta_1": lat.defect(W_star, 1), "delta_d": lat.defect(W_star, d)})
    t_end = pd.DataFrame(end_rows)

    # Curvature of the coupling correction (the convex/concave wording issue)
    curv = [{"d": d, "eps": e, "n": n, "d2R_du2": th.correction_curvature(d, e, n),
             "concave": th.correction_curvature(d, e, n) < 0}
            for d in sorted({cfg.d, *cfg.dims}) for e in (0.1, 0.4) for n in (100, 4000)]
    t_curv = pd.DataFrame(curv)

    # ---- audit of the constants actually printed in the paper
    audit = []

    def _chk(where, quantity, printed, formula, tol=0.01):
        rel = abs(printed - formula) / max(abs(formula), 1e-12)
        audit.append({"location": where, "quantity": quantity, "printed": printed,
                      "formula_value": formula, "rel_error": rel, "ok": rel <= tol})

    for e, printed in zip([0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40, 0.57, 0.80],
                          [13200, 6735, 3300, 1684, 825, 421, 206, 102, 52]):
        _chk("Table 2", f"predicted n* at eps={e}", float(printed),
             float(th.n_star(12, 0.5, e)))
    for dd, printed in zip([8, 12, 16], [280, 825, 1500]):
        _chk("Table 5", f"predicted n* at d={dd}, eps=0.2", float(printed),
             float(th.n_star(dd, 0.5, 0.2)))
    _chk("Sec. 4.3", "d(d-1) ratio d=16 over d=8", 4.29, (16 * 15) / (8 * 7))
    for k, printed in zip([1, 2, 3, 4, 6, 12],
                          [0.0, 0.5455, 0.7273, 0.8182, 0.9091, 1.0]):
        _chk("Table 1", f"gamma_{k} closed form", printed, float(th.gamma_generic(12, k)),
             tol=0.01 if printed else 1e-9)
    lo, hi = th.graded_window(12, 4, 0.35, 0.08, 0.5)
    _chk("Sec. 3 / Prop. 7", "graded window lower end", 80.0, lo, tol=0.25)
    _chk("Sec. 3 / Prop. 7", "graded window upper end", 2500.0, hi, tol=0.25)
    t_audit = pd.DataFrame(audit)

    summary = {
        "paper_constants_checked": int(len(t_audit)),
        "paper_constants_ok": int(t_audit.ok.sum()),
        "paper_constants_mismatched": t_audit[~t_audit.ok][
            ["location", "quantity", "printed", "formula_value"]].to_dict("records"),
        "prop1_param_count_exact": bool((t_prop1.p_k_formula == t_prop1.p_k_rank).all()),
        "prop1_max_projector_err": float(t_prop1[[
            "basis_orthonormal_err", "orbit_vs_group_average_err",
            "idempotent_err", "equivariance_err"]].to_numpy().max()),
        "prop2_monotone_all": bool(t_prop2.ok.all()),
        "prop2_pairs_checked": int(len(t_prop2)),
        "delta_1_max": float(t_end.delta_1.abs().max()),
        "delta_d_matches_eps": bool(np.allclose(t_end.delta_d, t_end.eps, atol=1e-10)),
        "coupling_correction_concave_everywhere": bool(t_curv.concave.all()),
    }
    return {"theory_checks_prop1": t_prop1, "theory_checks_prop2": t_prop2,
            "theory_checks_endpoints": t_end, "theory_checks_curvature": t_curv,
            "paper_constants_audit": t_audit}, summary


# -------------------------------------------------------- 2. generic defect


def run_generic_defect(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """
    Table 1.  Unlike the phase-diagram runs, here the defect direction ``R`` is
    **resampled** per seed, which is what tests the ensemble statement of
    Proposition 4 (Appendix E).
    """
    d, eps = cfg.d, cfg.table1_eps
    lat = lattice(d)
    vals = {k: [] for k in lat.ks}
    for s in range(cfg.table1_seeds):
        prof = generic_target(d, eps, seed=20_000 + s, lat=lat).defect_profile()
        for k in lat.ks:
            vals[k].append(prof[k] / eps**2)

    # A high-precision ensemble as well, so the closed form itself is tested
    # independently of how noisy a 40-seed average happens to be.
    hp_n = max(50 * cfg.table1_seeds, 500)
    hp = {k: [] for k in lat.ks}
    for s in range(hp_n):
        prof = generic_target(d, eps, seed=400_000 + s, lat=lat).defect_profile()
        for k in lat.ks:
            hp[k].append(prof[k] / eps**2)

    rows = []
    for k in lat.ks:
        v, h = np.array(vals[k]), np.array(hp[k])
        closed = float(th.gamma_generic(d, k))
        sem = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        hp_sem = float(h.std(ddof=1) / np.sqrt(len(h))) if len(h) > 1 else 0.0
        rows.append({
            "k": k, "p_k": lat.p[k], "gamma_closed_form": closed,
            "gamma_measured": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "sem": sem,
            "abs_dev": abs(float(v.mean()) - closed),
            "rel_dev_pct": 100 * abs(float(v.mean()) - closed) / closed if closed > 0 else 0.0,
            "z_score": abs(float(v.mean()) - closed) / sem if sem > 0 else 0.0,
            "gamma_measured_highprec": float(h.mean()), "highprec_draws": hp_n,
            "highprec_sem": hp_sem,
            "highprec_rel_dev_pct": 100 * abs(float(h.mean()) - closed) / closed if closed > 0 else 0.0,
            "highprec_z": abs(float(h.mean()) - closed) / hp_sem if hp_sem > 0 else 0.0,
        })
    t = pd.DataFrame(rows)
    summary = {
        "table1_eps": eps, "table1_seeds": cfg.table1_seeds,
        "max_abs_deviation": float(t.abs_dev.max()),
        "max_rel_deviation_pct": float(t.rel_dev_pct.max()),
        "argmax_rel_dev_k": int(t.loc[t.rel_dev_pct.idxmax(), "k"]),
        "max_z_score": float(t.z_score.max()),
        "consistent_within_3sem": bool(t.z_score.max() < 3.0),
        "highprec_draws": hp_n,
        "highprec_max_rel_deviation_pct": float(t.highprec_rel_dev_pct.max()),
        "highprec_max_z": float(t.highprec_z.max()),
        "highprec_within_1pct": bool(t.highprec_rel_dev_pct.max() < 1.0),
    }
    return {"table1_generic_defect": t}, summary


# --------------------------------------------------------- 3. phase diagram


def _phase_diagram(d: int, n_grid, eps_grid, seeds, sigma: float,
                   store: CheckpointStore, tag: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    lat = lattice(d)
    ks = lat.ks
    long_rows, cell_rows = [], []
    for eps in eps_grid:
        risks, target = _risk_curves(d, eps, seeds, n_grid, sigma, ks, store, tag)
        delta2 = {k: lat.defect(target.W, k) ** 2 for k in ks}
        mean = np.nanmean(risks, axis=0)
        std = np.nanstd(risks, axis=0, ddof=1)
        for ni, n in enumerate(n_grid):
            for kj, k in enumerate(ks):
                long_rows.append({
                    "d": d, "eps": eps, "n": int(n), "k": k, "p_k": lat.p[k],
                    "risk_mean": mean[ni, kj], "risk_std": std[ni, kj],
                    "delta2_k": delta2[k],
                    "risk_predicted_fs": float(th.risk_finite_sample(delta2[k], d, k, sigma, n)),
                    "risk_predicted_lo": float(th.risk_leading_order(delta2[k], d, k, sigma, n)),
                })
            valid = ~np.isnan(mean[ni])
            k_meas = int(np.array(ks)[valid][np.nanargmin(mean[ni][valid])])
            pred_fs = {k: float(th.risk_finite_sample(delta2[k], d, k, sigma, n)) for k in ks
                       if n * d >= lat.p[k] + 2}
            pred_lo = {k: float(th.risk_leading_order(delta2[k], d, k, sigma, n)) for k in ks
                       if n * d >= lat.p[k] + 2}
            cell_rows.append({
                "d": d, "eps": eps, "n": int(n),
                "k_measured": k_meas,
                "k_pred_finite_sample": th.best_k(pred_fs),
                "k_pred_leading_order": th.best_k(pred_lo),
                "endpoint": k_meas in (1, d),
                "agree_fs": k_meas == th.best_k(pred_fs),
                "agree_lo": k_meas == th.best_k(pred_lo),
            })
    long_df, cells = pd.DataFrame(long_rows), pd.DataFrame(cell_rows)
    summary = {
        "cells": int(len(cells)),
        "agree_finite_sample": int(cells.agree_fs.sum()),
        "agree_leading_order": int(cells.agree_lo.sum()),
        "endpoint_cells": int(cells.endpoint.sum()),
        "interior_cells": int((~cells.endpoint).sum()),
        "k_values_selected": sorted(int(x) for x in cells.k_measured.unique()),
    }
    return long_df, cells, summary


def run_phase_diagram(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """Section 4.1: the phase diagram at ``d = cfg.d``."""
    long_df, cells, s = _phase_diagram(cfg.d, cfg.phase_n, cfg.phase_eps, cfg.seeds,
                                       cfg.sigma, store, "phase")
    s["d"] = cfg.d
    s["theorem5_all_endpoints"] = s["interior_cells"] == 0
    return {f"phase_diagram_d{cfg.d}": long_df, f"phase_cells_d{cfg.d}": cells}, s


# -------------------------------------------------------------- 4. crossover


def _crossover_for_d(d: int, cfg: Config, eps_grid, seeds, n_points: int,
                     store: CheckpointStore) -> tuple[pd.DataFrame, dict]:
    lat = lattice(d)
    n_grid = cfg.cross_n_grid(n_points)
    rows = []
    for eps in eps_grid:
        risks, target = _risk_curves(d, eps, seeds, n_grid, cfg.sigma, ks=[1, d])
        per_seed = np.array([_first_crossing(risks[s, :, 0], risks[s, :, 1], n_grid)
                             for s in range(len(seeds))])
        med = float(np.nanmedian(per_seed))
        q1, q3 = (float(np.nanpercentile(per_seed, 25)), float(np.nanpercentile(per_seed, 75)))
        pred = float(th.n_star(d, cfg.sigma, eps))
        rows.append({
            "d": d, "eps": float(eps), "measured_n_star_median": med,
            "iqr_lo": q1, "iqr_hi": q3, "predicted_n_star": pred,
            "ratio": med / pred if pred > 0 else np.nan,
            "n_seeds_with_crossing": int(np.sum(np.isfinite(per_seed))),
        })
    t = pd.DataFrame(rows)
    fit = _fit_exponent(t.eps.values, t.measured_n_star_median.values)
    grid_spacing = float(np.exp(np.diff(np.log(n_grid)).mean()))
    iqr_within_grid = bool(((t.iqr_hi / np.maximum(t.iqr_lo, 1)) <= grid_spacing**2 + 1e-9).all())
    summary = {
        "d": d, **{f"exponent_{k}": v for k, v in fit.items()},
        "predicted_exponent": -2.0,
        "exponent_ci_contains_minus2": bool(fit["ci95"][0] <= -2.0 <= fit["ci95"][1]),
        "max_ratio": float(t.ratio.max()), "min_ratio": float(t.ratio.min()),
        "median_ratio": float(t.ratio.median()),
        "grid_log_spacing": grid_spacing,
        "iqr_within_two_grid_steps": iqr_within_grid,
    }
    return t, summary


def run_crossover(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """Section 4.2 and Table 2: locate ``n*`` and regress ``log n*`` on ``log eps``."""
    t, s = _crossover_for_d(cfg.d, cfg, cfg.cross_eps, cfg.seeds, cfg.cross_n_points, store)
    return {"table2_crossover": t}, s


# ------------------------------------------------------------- 5. dimension


def run_dimension(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """Section 4.3 and Table 5: ``d in {8, 16}`` phase diagrams and the ``d(d-1)`` ratio."""
    tables, per_d = {}, {}
    seeds = cfg.seeds[: max(cfg.dim_cross_seeds, 1)]

    # crossover at the main dimension too, with the same reduced protocol, so the
    # three rows of Table 5 are measured identically
    t_main, s_main = _crossover_for_d(cfg.d, cfg, cfg.dim_eps, seeds,
                                      cfg.dim_cross_n_points, store)
    tables[f"crossover_d{cfg.d}"] = t_main
    per_d[cfg.d] = {"phase": {}, "crossover": s_main, "table": t_main}

    for d in cfg.dims:
        long_df, cells, s = _phase_diagram(d, cfg.dim_n[d], cfg.dim_eps, cfg.seeds,
                                           cfg.sigma, store, "dim")
        tables[f"phase_diagram_d{d}"] = long_df
        tables[f"phase_cells_d{d}"] = cells
        tcross, scross = _crossover_for_d(d, cfg, cfg.dim_eps, seeds,
                                          cfg.dim_cross_n_points, store)
        tables[f"crossover_d{d}"] = tcross
        per_d[d] = {"phase": s, "crossover": scross, "table": tcross}

    # d(d-1) ratio at a common eps
    common_eps = cfg.dim_eps[len(cfg.dim_eps) // 2]
    ratio_rows = []
    for d, rec in per_d.items():
        row = rec["table"].iloc[(rec["table"].eps - common_eps).abs().argmin()]
        ratio_rows.append({"d": d, "eps": float(row.eps),
                           "measured_n_star": float(row.measured_n_star_median),
                           "predicted_n_star": float(row.predicted_n_star),
                           "ratio": float(row.ratio)})
    t_ratio = pd.DataFrame(ratio_rows).sort_values("d")
    tables["table5_dimension"] = t_ratio

    summary = {"eps_for_ratio": float(common_eps)}
    for d, rec in per_d.items():
        if rec["phase"]:
            summary[f"d{d}_cells"] = rec["phase"]["cells"]
            summary[f"d{d}_agree_fs"] = rec["phase"]["agree_finite_sample"]
            summary[f"d{d}_interior_cells"] = rec["phase"]["interior_cells"]
        summary[f"d{d}_exponent"] = rec["crossover"]["exponent_exponent"]
        summary[f"d{d}_exponent_stderr"] = rec["crossover"]["exponent_stderr"]
        summary[f"d{d}_ci_contains_minus2"] = rec["crossover"]["exponent_ci_contains_minus2"]
    if len(t_ratio) >= 2:
        hi, lo = t_ratio.iloc[-1], t_ratio.iloc[0]
        summary["measured_ratio_hi_over_lo"] = float(hi.measured_n_star / lo.measured_n_star)
        summary["predicted_ratio_hi_over_lo"] = float(
            (hi.d * (hi.d - 1)) / (lo.d * (lo.d - 1)))
        summary["ratio_pair"] = [int(lo.d), int(hi.d)]
    return tables, summary


# ---------------------------------------------------------------- 6. graded


def run_graded(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """Section 4.4 and Proposition 7: the ladder ``12 -> 4 -> 1`` and the window (8)."""
    d, k0, b, c = cfg.d, cfg.graded_k0, cfg.graded_b, cfg.graded_c
    lat = lattice(d)
    n_grid = cfg.graded_n
    risks, target = _risk_curves(d, float("nan"), cfg.seeds, n_grid, cfg.sigma, lat.ks,
                                 store, "graded", kind="graded", k0=k0, b=b, c=c)
    delta2 = {k: lat.defect(target.W, k) ** 2 for k in lat.ks}

    # Defect profile.  The closed forms are expectations over the random unit
    # components u_{k0}, u_1, so the like-for-like comparison is an ensemble over
    # resampled graded targets (the same distinction the paper draws in App. E
    # between Table 1 and Sections 4.1-4.2).  The fixed target actually used for
    # the ladder is reported alongside.
    ens = {k: [] for k in lat.ks}
    for s_ in range(max(cfg.table1_seeds, 20)):
        prof = graded_target(d, k0, b, c, seed=30_000 + s_, lat=lat).defect_profile()
        for k in lat.ks:
            ens[k].append(prof[k])
    ens_mean = {k: float(np.mean(v)) for k, v in ens.items()}

    prof_rows = [{
        "k": k, "delta2_measured": ens_mean[k], "delta2_fixed_target": delta2[k],
        "delta2_prop7": float(th.delta2_graded(d, k, k0, b, c)),
        "delta2_exact": float(th.delta2_graded_exact(d, k, k0, b, c)),
    } for k in lat.ks]
    t_prof = pd.DataFrame(prof_rows)
    for col, src in (("rel_err_prop7_pct", "delta2_prop7"), ("rel_err_exact_pct", "delta2_exact")):
        t_prof[col] = 100 * (t_prof[src] - t_prof.delta2_measured).abs() / \
            t_prof.delta2_measured.replace(0, np.nan)

    mean = np.nanmean(risks, axis=0)
    rows = []
    for ni, n in enumerate(n_grid):
        valid = ~np.isnan(mean[ni])
        k_meas = int(np.array(lat.ks)[valid][np.nanargmin(mean[ni][valid])])
        pred = {k: float(th.risk_finite_sample(delta2[k], d, k, cfg.sigma, n))
                for k in lat.ks if n * d >= lat.p[k] + 2}
        rows.append({"n": int(n), "k_measured": k_meas, "k_predicted": th.best_k(pred),
                     "agree": k_meas == th.best_k(pred),
                     **{f"risk_k{k}": mean[ni, j] for j, k in enumerate(lat.ks)}})
    t_ladder = pd.DataFrame(rows)

    lo, hi = th.graded_window(d, k0, b, c, cfg.sigma)
    sel = t_ladder[t_ladder.k_measured == k0]
    summary = {
        "d": d, "k0": k0, "b": b, "c": c,
        "defect_measured_fixed_target": {int(k): float(delta2[k]) for k in lat.ks},
        "defect_measured_ensemble": {int(k): float(ens_mean[k]) for k in lat.ks},
        "prop7_max_rel_err_pct": float(np.nanmax(t_prof.rel_err_prop7_pct.values)),
        "exact_max_rel_err_pct": float(np.nanmax(t_prof.rel_err_exact_pct.values)),
        "ladder": list(dict.fromkeys(int(x) for x in t_ladder.k_measured)),
        "agree_cells": int(t_ladder.agree.sum()), "cells": int(len(t_ladder)),
        "measured_window": [int(sel.n.min()), int(sel.n.max())] if len(sel) else None,
        "predicted_window": [float(lo), float(hi)],
        "interior_optimum_observed": bool(len(sel) > 0),
        "intermediate_decades": float(np.log10(sel.n.max() / sel.n.min())) if len(sel) > 1 else 0.0,
    }
    return {"graded_defect_profile": t_prof, "graded_ladder": t_ladder}, summary


# ------------------------------------------------------------- 7. nonlinear


def run_nonlinear(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """Section 4.5 and Table 3: the two-layer tanh control."""
    d = cfg.d
    lat = lattice(d)
    rows = []
    for eps in cfg.nl_eps:
        for seed in range(cfg.nl_seeds):
            rng = np.random.default_rng(900_000 + seed)
            teacher = make_teacher(d, eps, rng, lat)
            for n in cfg.nl_n:
                for k in cfg.nl_ks:
                    r = train_projected(
                        teacher, k, int(n), cfg.nl_sigma,
                        np.random.default_rng(hash((seed, n, k)) % (2**32)),
                        lat, steps=cfg.nl_steps, lr=cfg.nl_lr, width=cfg.nl_width,
                        n_test=cfg.nl_test, eps=eps, seed=seed,
                        keep_weights=(seed == 0 and n == cfg.nl_n[-1]),
                    )
                    if r.W1 is not None:
                        store.save_model(f"nonlinear_eps{eps:g}_n{n}_k{k}_W1", r.W1, r.W1,
                                         layer=1, k=k, n=int(n), eps=float(eps), seed=seed,
                                         steps=cfg.nl_steps, test_mse=r.test_mse)
                        store.save_model(f"nonlinear_eps{eps:g}_n{n}_k{k}_W2", r.W2, r.W2,
                                         layer=2, k=k, n=int(n), eps=float(eps), seed=seed,
                                         steps=cfg.nl_steps, test_mse=r.test_mse)
                    rows.append({"eps": eps, "seed": seed, "n": int(n), "k": k,
                                 "test_mse": r.test_mse, "train_mse": r.train_mse})
    raw = pd.DataFrame(rows)
    agg = raw.groupby(["eps", "n", "k"]).test_mse.agg(["mean", "std"]).reset_index()
    wide = agg.pivot_table(index=["eps", "n"], columns="k", values="mean")
    wide_sd = agg.pivot_table(index=["eps", "n"], columns="k", values="std")
    table3 = wide.join(wide_sd, lsuffix="_mean", rsuffix="_sd").reset_index()

    # crossover of the equivariant model against the dense one
    cross = {}
    kmax, kmin = max(cfg.nl_ks), min(cfg.nl_ks)
    for eps in cfg.nl_eps:
        sub = wide.loc[eps]
        wins = sub[kmax] < sub[kmin]
        if wins.all():
            cross[str(eps)] = None                      # equivariant wins everywhere
        elif (~wins).all():
            cross[str(eps)] = float(sub.index.min())
        else:
            cross[str(eps)] = float(sub.index[np.argmax(~wins.values)])

    best = agg.loc[agg.groupby(["eps", "n"]).test_mse_mean.idxmin()] if "test_mse_mean" in agg \
        else agg.loc[agg.groupby(["eps", "n"])["mean"].idxmin()]
    summary = {
        "seeds": cfg.nl_seeds, "steps": cfg.nl_steps, "n_grid": cfg.nl_n, "ks": cfg.nl_ks,
        "crossover_n_by_eps": cross,
        "crossover_moves_with_eps": bool(
            len(cfg.nl_eps) >= 2 and (
                (cross[str(cfg.nl_eps[0])] is None and cross[str(cfg.nl_eps[-1])] is not None)
                or (cross[str(cfg.nl_eps[0])] is not None and cross[str(cfg.nl_eps[-1])] is not None
                    and cross[str(cfg.nl_eps[-1])] <= cross[str(cfg.nl_eps[0])])
            )),
        "best_k_by_cell": {f"eps{r.eps:g}_n{int(r.n)}": int(r.k) for r in best.itertuples()},
    }
    return {"table3_nonlinear": table3, "nonlinear_raw": raw}, summary


# ------------------------------------------- 8. variance constant / coupling


def run_variance_constant(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """
    Appendix C / F: the ratio ``rho_k = n (risk - delta^2 - delta^2 p_k / n) / (sigma^2 p_k)``,
    and -- going beyond the paper -- a **direct fit of the coupling coefficient**.

    Proposition 3 asserts the coupling term is ``delta_k^2 tr(Pi_k)/n`` with
    ``tr(Pi_k) = p_k``.  Here we estimate ``c_k`` in

        E risk = delta_k^2 + sigma^2 p_k / n + c_k delta_k^2 / n

    by least squares on ``n (risk - delta^2 - sigma^2 p_k / n) / delta^2`` across
    sample sizes, with enough seeds to resolve it.
    """
    d, sigma = cfg.d, cfg.sigma
    lat = lattice(d)

    # ---- rho_k across defect sizes (Table 4 of the paper)
    rows = []
    for eps in cfg.var_eps:
        risks, target = _risk_curves(d, eps, cfg.seeds, [cfg.var_n], sigma, lat.ks)
        for kj, k in enumerate(lat.ks):
            d2 = lat.defect(target.W, k) ** 2
            p, n = lat.p[k], cfg.var_n
            r = risks[:, 0, kj]
            rho = n * (r - d2 - d2 * p / n) / (sigma**2 * p)
            rho_lo = n * (r - d2) / (sigma**2 * p)
            rows.append({"eps": eps, "k": k, "p_k": p, "n": n, "delta2": d2,
                         "rho_paper_mean": float(np.nanmean(rho)),
                         "rho_paper_sd": float(np.nanstd(rho, ddof=1)),
                         "rho_no_coupling_mean": float(np.nanmean(rho_lo)),
                         "rho_no_coupling_sd": float(np.nanstd(rho_lo, ddof=1))})
    t_rho = pd.DataFrame(rows)

    # ---- direct fit of the coupling coefficient c_k
    eps = cfg.coupling_eps
    seeds = list(range(50_000, 50_000 + cfg.coupling_seeds))
    risks, target = _risk_curves(d, eps, seeds, cfg.coupling_n, sigma, lat.ks)
    crows = []
    for kj, k in enumerate(lat.ks):
        d2 = lat.defect(target.W, k) ** 2
        if d2 <= 0:
            continue
        p = lat.p[k]
        ns = np.array(cfg.coupling_n, dtype=float)
        rbar = np.nanmean(risks[:, :, kj], axis=0)
        rsem = np.nanstd(risks[:, :, kj], axis=0, ddof=1) / np.sqrt(len(seeds))
        chat = ns * (rbar - d2 - sigma**2 * p / ns) / d2
        csem = ns * rsem / d2
        w = 1.0 / np.maximum(csem, 1e-12) ** 2
        c_fit = float(np.sum(w * chat) / np.sum(w))
        c_err = float(np.sqrt(1.0 / np.sum(w)))
        crows.append({"k": k, "p_k": p, "delta2": d2, "c_measured": c_fit, "c_stderr": c_err,
                      "c_paper_p_k": float(p), "c_over_paper": c_fit / p,
                      "c_candidate_d_plus_1_over_k": (d + 1) / k,
                      "c_over_candidate": c_fit / ((d + 1) / k),
                      **{f"c_at_n{int(n)}": float(v) for n, v in zip(cfg.coupling_n, chat)}})
    t_coup = pd.DataFrame(crows)

    summary = {
        "rho_range_at_small_eps": [
            float(t_rho[t_rho.eps == min(cfg.var_eps)].rho_paper_mean.min()),
            float(t_rho[t_rho.eps == min(cfg.var_eps)].rho_paper_mean.max())],
        "rho_within_0p97_1p04_at_small_eps": bool(
            t_rho[t_rho.eps == min(cfg.var_eps)].rho_paper_mean.between(0.97, 1.04).all()),
        "rho_no_coupling_range_at_small_eps": [
            float(t_rho[t_rho.eps == min(cfg.var_eps)].rho_no_coupling_mean.min()),
            float(t_rho[t_rho.eps == min(cfg.var_eps)].rho_no_coupling_mean.max())],
        "coupling_eps": eps, "coupling_seeds": len(seeds),
        "coupling_measured": {int(r.k): float(r.c_measured) for r in t_coup.itertuples()},
        "coupling_paper": {int(r.k): float(r.c_paper_p_k) for r in t_coup.itertuples()},
        "coupling_ratio_measured_over_paper": {
            int(r.k): float(r.c_over_paper) for r in t_coup.itertuples()},
        "coupling_candidate_d_plus_1_over_k": {
            int(r.k): float(r.c_candidate_d_plus_1_over_k) for r in t_coup.itertuples()},
        "coupling_ratio_measured_over_candidate": {
            int(r.k): float(r.c_over_candidate) for r in t_coup.itertuples()},
        "coupling_consistent_with_paper": bool(
            len(t_coup) and np.all(np.abs(t_coup.c_measured - t_coup.c_paper_p_k)
                                   < 3 * t_coup.c_stderr + 0.05 * t_coup.c_paper_p_k)),
    }
    return {"table4_variance_constant": t_rho, "coupling_fit": t_coup}, summary


# ------------------------------------------------------------ 9. non-abelian


def run_nonabelian(cfg: Config, store: CheckpointStore) -> tuple[dict, dict]:
    """Appendix H / Proposition 8."""
    rows, audits = [], {}
    for name in cfg.nonabelian_groups:
        for rep_kind in ("regular", "natural"):
            a = audit_group(name, GROUPS[name], rep_kind=rep_kind)
            scan = interior_optimum_scan(a)
            audits[f"{name}_{rep_kind}"] = {"audit": {k: v for k, v in a.items() if k != "rows"},
                                            "scan": scan}
            for r in a["rows"]:
                rows.append({"group": name, "rep": rep_kind, "group_order": a["group_order"],
                             "D": a["D"], "abelian": a["abelian"], **r})
    t = pd.DataFrame(rows)

    reg = t[t.rep == "regular"]
    nat = t[t.rep == "natural"]
    bad_paper = t[(t.p_H_numeric > 0) & (np.abs(t.p_H_paper_formula - t.p_H_numeric) > 1e-6)]

    summary = {
        "groups": cfg.nonabelian_groups,
        "regular_rep_formula_pG_equals_D2_over_H": bool(
            np.allclose(reg.p_H_trace, reg.D**2 / reg.order)),
        "regular_rep_affine_for_all_groups": bool(
            all(v["audit"]["affine"] for k, v in audits.items() if k.endswith("regular"))),
        "regular_rep_collapse_holds_including_nonabelian": bool(
            all(v["scan"]["interior_cells"] == 0
                for k, v in audits.items() if k.endswith("regular"))),
        "natural_rep_affine": {k.split("_")[0]: bool(v["audit"]["affine"])
                               for k, v in audits.items() if k.endswith("natural")},
        "natural_rep_interior_cells": {k.split("_")[0]: v["scan"]["interior_cells"]
                                       for k, v in audits.items() if k.endswith("natural")},
        "paper_pH_formula_disagreements": int(len(bad_paper)),
        "paper_pH_formula_example": (
            bad_paper[["group", "rep", "order", "decomposition",
                       "p_H_numeric", "p_H_paper_formula"]].head(5).to_dict("records")
            if len(bad_paper) else []),
        "numeric_vs_trace_max_err": float(
            np.abs(t[t.p_H_numeric > 0].p_H_numeric - t[t.p_H_numeric > 0].p_H_trace).max())
        if (t.p_H_numeric > 0).any() else 0.0,
    }
    return {"nonabelian_audit": t}, summary


STAGES = {
    "theory_checks": run_theory_checks,
    "generic_defect": run_generic_defect,
    "phase_diagram": run_phase_diagram,
    "crossover": run_crossover,
    "dimension": run_dimension,
    "graded": run_graded,
    "nonlinear": run_nonlinear,
    "variance_constant": run_variance_constant,
    "nonabelian": run_nonabelian,
}
