#!/usr/bin/env python3
"""
A short worked example: build the lattice, make a target, fit every architecture,
and compare the measured minimiser against the closed-form prediction.

    python scripts/example_walkthrough.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from symbudget import theory as th
from symbudget.data import StreamingStats, generic_target, graded_target
from symbudget.estimators import fit_all_k
from symbudget.lattice import lattice

d, sigma = 12, 0.5
lat = lattice(d)

print(f"lattice of C_{d}: k = {lat.ks}")
print(f"parameter counts: p_k = {[lat.p[k] for k in lat.ks]}")
print()

# ---- 1. where is the phase boundary?
for eps in (0.05, 0.2, 0.8):
    print(f"eps = {eps:<5g}  n* = sigma^2 d(d-1)/eps^2 = {float(th.n_star(d, sigma, eps)):9.1f}")
print()

# ---- 2. a generic defect: the lattice collapses to its endpoints
eps = 0.2
target = generic_target(d, eps, seed=0, lat=lat)
delta2 = {k: lat.defect(target.W, k) ** 2 for k in lat.ks}
print(f"generic defect, eps = {eps}")
print("  k   p_k   delta^2    risk(n=200)  risk(n=4000)")
curves = {}
for n in (200, 4000):
    ss = StreamingStats(target.W, sigma, np.random.default_rng(0)).advance_to(n)
    S, C, nn = ss.stats
    curves[n] = {k: r.excess_risk for k, r in fit_all_k(S, C, nn, target.W, lat).items()}
for k in lat.ks:
    print(f" {k:3d} {lat.p[k]:5d}  {delta2[k]:.6f}   {curves[200][k]:.6f}     {curves[4000][k]:.6f}")
for n in (200, 4000):
    best = min(curves[n], key=curves[n].get)
    pred = th.RiskModel(d, sigma, delta2).argmin(n)
    print(f"  n = {n:5d}: measured k* = {best:2d}, predicted k* = {pred:2d}  "
          f"(endpoint: {best in (1, d)})")
print()

# ---- 3. a graded defect: the lattice reopens
k0, b, c = 4, 0.35, 0.08
gt = graded_target(d, k0, b, c, seed=7, lat=lat)
g_delta2 = {k: lat.defect(gt.W, k) ** 2 for k in lat.ks}
print(f"graded defect, k0 = {k0}, (b, c) = ({b}, {c})")
print("  k   measured    eq.(7)      exact")
for k in lat.ks:
    print(f" {k:3d}  {g_delta2[k]:.6f}   {float(th.delta2_graded(d, k, k0, b, c)):.6f}"
          f"   {float(th.delta2_graded_exact(d, k, k0, b, c)):.6f}")
lo, hi = th.graded_window(d, k0, b, c, sigma)
print(f"  predicted window for k* = {k0}:  {lo:.0f} <~ n <~ {hi:.0f}")

model = th.RiskModel(d, sigma, g_delta2)
ladder = []
for n in np.geomspace(20, 40000, 30):
    k = model.argmin(float(n))
    if not ladder or ladder[-1] != k:
        ladder.append(k)
print(f"  predicted ladder as n grows: {' -> '.join(map(str, ladder))}")
