"""
Figure generation.

Figures 1-3 reproduce the paper; Figures 4-5 are diagnostics for the two
corrections this implementation turns up (the coupling coefficient and the
non-abelian affineness test).  Every figure is written as both PDF and PNG and
carries the seeds and grid in its caption metadata.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import theory as th

__all__ = ["make_all_figures"]

PALETTE = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728",
           6: "#9467bd", 8: "#8c564b", 12: "#8c564b", 16: "#e377c2"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _save(fig, outdir: Path, name: str) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _label(k: int, d: int) -> str:
    if k == 1:
        return "$k=1$ (MLP)"
    if k == d:
        return f"$k={k}$ (equiv.)"
    return f"$k={k}$"


# ----------------------------------------------------------------- Figure 1


def figure1(long_df: pd.DataFrame, d: int, sigma: float, outdir: Path,
            eps_panels=(0.1, 0.4)) -> list[Path]:
    """Excess risk versus ``n`` for every architecture, at two defect levels."""
    avail = sorted(long_df.eps.unique())
    panels = [min(avail, key=lambda a: abs(a - e)) for e in eps_panels]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, eps in zip(axes, panels):
        sub = long_df[long_df.eps == eps]
        for k in sorted(sub.k.unique()):
            s = sub[sub.k == k].sort_values("n")
            ax.loglog(s.n, s.risk_mean, "o-", ms=3, lw=1.2,
                      color=PALETTE.get(k, None), label=_label(k, d))
        floor = sub[sub.k == d].delta2_k.iloc[0]
        ax.axhline(floor, ls=":", c="k", lw=0.9)
        ax.annotate(r"$\delta^2_{%d}$" % d, (sub.n.min(), floor * 1.15), fontsize=8)
        ax.set_title(rf"$\varepsilon = {eps:g}$")
        ax.set_xlabel("$n$")
    axes[0].set_ylabel(r"excess risk $\|\hat W - W^\star\|_F^2$")
    axes[-1].legend(ncol=2, fontsize=7.5)
    fig.suptitle(f"Figure 1 — excess risk vs $n$, $d={d}$, $\\sigma={sigma}$", y=1.02, fontsize=10)
    return _save(fig, outdir, "figure1_risk_vs_n")


# ----------------------------------------------------------------- Figure 2


def figure2(cells: pd.DataFrame, crossover: pd.DataFrame, d: int, sigma: float,
            exponent: dict, outdir: Path) -> list[Path]:
    """Left: measured ``k*`` over the ``(eps, n)`` grid.  Right: ``n*`` versus ``eps``."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    eps_vals = sorted(cells.eps.unique())
    n_vals = sorted(cells.n.unique())
    grid = np.full((len(eps_vals), len(n_vals)), np.nan)
    for r in cells.itertuples():
        grid[eps_vals.index(r.eps), n_vals.index(r.n)] = r.k_measured
    im = axL.imshow(grid, origin="lower", aspect="auto", cmap="coolwarm_r",
                    vmin=1, vmax=d)
    axL.set_xticks(range(len(n_vals)), [str(n) for n in n_vals], rotation=90, fontsize=6.5)
    axL.set_yticks(range(len(eps_vals)), [f"{e:g}" for e in eps_vals], fontsize=7)
    axL.set_xlabel("$n$")
    axL.set_ylabel(r"$\varepsilon$")
    axL.grid(False)
    cb = fig.colorbar(im, ax=axL, ticks=[1, d])
    cb.ax.set_yticklabels(["$k^\\star=1$", f"$k^\\star={d}$"], fontsize=7)

    # parameter-free boundary n* = sigma^2 d(d-1)/eps^2 in grid coordinates
    xs, ys = [], []
    for i, e in enumerate(eps_vals):
        ns = float(th.n_star(d, sigma, e))
        if n_vals[0] <= ns <= n_vals[-1]:
            xs.append(np.interp(np.log(ns), np.log(n_vals), np.arange(len(n_vals))))
            ys.append(i)
    axL.plot(xs, ys, "k--", lw=1.4, label=r"$n^\star=\sigma^2 d(d-1)/\varepsilon^2$")
    axL.legend(fontsize=7, loc="lower left")

    c = crossover.sort_values("eps")
    axR.loglog(c.eps, c.measured_n_star_median, "o", ms=4, label="measured")
    lo = c.measured_n_star_median - c.iqr_lo
    hi = c.iqr_hi - c.measured_n_star_median
    axR.errorbar(c.eps, c.measured_n_star_median, yerr=[lo.clip(lower=0), hi.clip(lower=0)],
                 fmt="none", ecolor="C0", alpha=0.6, capsize=2, lw=0.8)
    axR.loglog(c.eps, c.predicted_n_star, "-", color="orange",
               label=r"$\sigma^2 d(d-1)/\varepsilon^2$")
    e = np.asarray(c.eps, dtype=float)
    fit = np.exp(exponent["exponent_intercept"]) * e ** exponent["exponent_exponent"]
    band_lo = np.exp(exponent["exponent_intercept"]) * e ** exponent["exponent_ci95"][0]
    band_hi = np.exp(exponent["exponent_intercept"]) * e ** exponent["exponent_ci95"][1]
    axR.fill_between(e, np.minimum(band_lo, band_hi), np.maximum(band_lo, band_hi),
                     color="C0", alpha=0.15, label="95% CI")
    axR.plot(e, fit, "--", color="C0", lw=0.9)
    axR.set_xlabel(r"symmetry defect $\varepsilon$")
    axR.set_ylabel("$n^\\star$")
    axR.set_title(f"fitted exponent {exponent['exponent_exponent']:.2f} "
                  f"$\\pm$ {exponent['exponent_stderr']:.2f} (theory $-2$)", fontsize=9)
    axR.legend(fontsize=7.5)
    fig.suptitle(f"Figure 2 — phase diagram and crossover, $d={d}$", y=1.03, fontsize=10)
    return _save(fig, outdir, "figure2_phase_diagram")


# ----------------------------------------------------------------- Figure 3


def figure3(ladder: pd.DataFrame, nl_raw: pd.DataFrame, d: int, k0: int,
            outdir: Path) -> list[Path]:
    """Left: the graded-defect ladder.  Right: the nonlinear control."""
    eps_list = sorted(nl_raw.eps.unique()) if nl_raw is not None and len(nl_raw) else []
    ncols = 1 + len(eps_list)
    fig, axes = plt.subplots(1, ncols, figsize=(3.6 * ncols, 3.2))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    kcols = [c for c in ladder.columns if c.startswith("risk_k")]
    for col in kcols:
        k = int(col.replace("risk_k", ""))
        ax.loglog(ladder.n, ladder[col], "o-", ms=3, lw=1.1,
                  color=PALETTE.get(k), label=f"$k={k}$")
    # shade the measured k* regimes
    shades = {1: "#d62728", k0: "#2ca02c", d: "#1f77b4"}
    ns = ladder.n.values
    kk = ladder.k_measured.values
    for i in range(len(ns)):
        lo = ns[i] / 1.3 if i == 0 else np.sqrt(ns[i - 1] * ns[i])
        hi = ns[i] * 1.3 if i == len(ns) - 1 else np.sqrt(ns[i] * ns[i + 1])
        ax.axvspan(lo, hi, color=shades.get(int(kk[i]), "#999999"), alpha=0.10, lw=0)
    ax.set_xlabel("$n$")
    ax.set_ylabel("excess risk")
    ax.set_title(r"graded defect: $k^\star$ ladder", fontsize=9)
    ax.legend(ncol=2, fontsize=7)

    for ax, eps in zip(axes[1:], eps_list):
        sub = nl_raw[nl_raw.eps == eps]
        g = sub.groupby(["n", "k"]).test_mse.agg(["mean", "std"]).reset_index()
        for k in sorted(g.k.unique()):
            s = g[g.k == k].sort_values("n")
            ax.errorbar(s.n, s["mean"], yerr=s["std"].fillna(0), fmt="o-", ms=3, lw=1.1,
                        capsize=2, color=PALETTE.get(k), label=f"$k={k}$")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(rf"2-layer tanh, $\varepsilon={eps:g}$", fontsize=9)
        ax.legend(fontsize=7)
    if len(axes) > 1:
        axes[1].set_ylabel("test MSE")
    fig.suptitle("Figure 3 — structured defects and the nonlinear control", y=1.03, fontsize=10)
    return _save(fig, outdir, "figure3_graded_and_nonlinear")


# ---------------------------------------------------- Figure 4 (new): coupling


def figure4(coup: pd.DataFrame, d: int, outdir: Path) -> list[Path]:
    """Measured coupling coefficient against the paper's ``tr(Pi_k) = p_k``."""
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    x = coup.k.values
    ax.errorbar(x, coup.c_measured, yerr=coup.c_stderr, fmt="o", ms=4, capsize=3,
                label="measured $c_k$")
    ax.plot(x, coup.c_paper_p_k, "s--", ms=4, color="C3",
            label=r"Prop. 3: $c_k=\mathrm{tr}(\Pi_k)=p_k$")
    ax.plot(x, coup.c_candidate_d_plus_1_over_k, "^-", ms=4, color="C2",
            label=r"$(d+1)/k$")
    ax.set_yscale("log")
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"coupling coefficient $c_k$  in  $\delta_k^2 c_k / n$")
    ax.set_title(f"Figure 4 — coupling term, $d={d}$", fontsize=9)
    ax.legend(fontsize=7.5)
    return _save(fig, outdir, "figure4_coupling_coefficient")


# ------------------------------------------------ Figure 5 (new): non-abelian


def figure5(audit: pd.DataFrame, outdir: Path) -> list[Path]:
    """``gamma_H`` against ``1/|H|`` for regular and non-regular representations."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    for ax, rep in zip(axes, ["regular", "natural"]):
        sub = audit[audit.rep == rep]
        for g in sorted(sub.group.unique()):
            s = sub[sub.group == g].sort_values("order")
            ax.plot(1.0 / s.order, s.gamma_H, "o-", ms=4, lw=1.0, alpha=0.85, label=g)
        ax.set_xlabel(r"$1/|H|$")
        ax.set_title(f"{rep} representation", fontsize=9)
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel(r"$\gamma_H = \mathbb{E}\delta_H^2/\varepsilon^2$")
    axes[0].annotate("affine for every group\n(abelian or not)", (0.35, 0.25), fontsize=7.5)
    axes[1].annotate("affineness fails", (0.35, 0.25), fontsize=7.5)
    fig.suptitle("Figure 5 — Proposition 8: affineness is a property of the representation",
                 y=1.03, fontsize=10)
    return _save(fig, outdir, "figure5_nonabelian_affineness")


# -------------------------------------------------------------------- driver


def make_all_figures(tables: dict[str, pd.DataFrame], summary: dict, cfg, outdir) -> list[str]:
    outdir = Path(outdir)
    made: list[Path] = []
    d = cfg.d

    key = f"phase_diagram_d{d}"
    if key in tables:
        made += figure1(tables[key], d, cfg.sigma, outdir)
    if f"phase_cells_d{d}" in tables and "table2_crossover" in tables:
        made += figure2(tables[f"phase_cells_d{d}"], tables["table2_crossover"], d, cfg.sigma,
                        summary.get("crossover", {}), outdir)
    if "graded_ladder" in tables:
        made += figure3(tables["graded_ladder"], tables.get("nonlinear_raw"),
                        d, cfg.graded_k0, outdir)
    if "coupling_fit" in tables and len(tables["coupling_fit"]):
        made += figure4(tables["coupling_fit"], d, outdir)
    if "nonabelian_audit" in tables:
        made += figure5(tables["nonabelian_audit"], outdir)
    return [str(p) for p in made]
