# Results of the shipped run

Preset `paper` (Appendix B protocol: `d ∈ {8, 12, 16}`, `σ = 0.5`, 40 seeds, 10 seeds for
the nonlinear control). Single core, no GPU. Total wall-clock about 3.5 minutes; the
nonlinear control is roughly 90% of it.

Everything below is read off `results/results.json` and `results/csv/`.

---

## Reproduced closely

**Table 2 — the crossover `n*(ε)` at `d = 12`.** Eight of the nine measured medians
match the paper's printed values exactly:

| ε | measured `n*` (this run) | paper | predicted `σ²d(d−1)/ε²` | ratio |
|---|---|---|---|---|
| 0.050 | 13849 | 13849 | 13200 | 1.05 |
| 0.071 | 7272 | 6393 (at ε = 0.07) | 6600 | 1.10 |
| 0.100 | 3357 | 3357 | 3300 | 1.02 |
| 0.141 | 1763 | 1763 | 1650 | 1.07 |
| 0.200 | 926 | 926 | 825 | 1.12 |
| 0.283 | 427 | 427 | 412.5 | 1.04 |
| 0.400 | 224 | 224 | 206.25 | 1.09 |
| 0.566 | 118 | 118 | 103.1 | 1.14 |
| 0.800 | 70 | 70 | 51.6 | 1.36 |

The single difference at the second grid point is a grid-labelling artefact: a nine-point
geometric grid on `[0.05, 0.8]` puts the point at 0.0707, which the paper prints as 0.07.
The `1.36` overshoot at `ε = 0.8` is reproduced to two decimals, including the paper's
explanation that it is where the finite-sample correction stops being negligible.

**Theorem 5.** 60/60 phase-diagram cells at `d = 12` have an endpoint minimiser; the only
`k*` values ever selected are `{1, 12}`. Measured and predicted `k*` agree on 59/60 (the
one disagreement is a cell on the boundary where the two endpoint risks differ by less
than the seed-to-seed spread).

**Proposition 4.** At 2000 resampled defect directions the measured `γ_k` sits within
0.31% of the closed form at every `k`; at 40 seeds the largest deviation is 2.7%, which is
1.6 standard errors — Monte-Carlo noise, not bias.

**Proposition 3 (leading order).** `ρ_k ∈ [0.978, 0.999]` at `n = 4000`, `ε = 0.05`,
against the paper's claimed `[0.97, 1.04]`.

**Proposition 7.** Ladder `12 → 4 → 1`, measured window `80 ≲ n ≲ 2500` — exactly the
interval the paper reports. Predicted and measured `k*` agree on 13 of 14 sample sizes,
the sole disagreement being `n = 50`, adjacent to the first boundary. That is the figure
the paper's earlier draft gives (the embedded earlier figure text in the submitted PDF
says "13 of 14 … the sole disagreement being the single grid point n = 50"); the final
text claims 14/14. This run supports 13/14.

**Nonlinear control.** At `ε = 0.1` the equivariant model wins across the whole range; at
`ε = 0.4` the crossover lands at `n ≈ 1600`. Direction and approximate location as
claimed.

**Dimension dependence.** 48/48 cells agree at `d = 16`; 31/32 at `d = 8`, with one
interior cell (again a near-boundary tie). Fitted crossover exponents `−1.96` (`d = 8`),
`−1.97` (`d = 12`), `−1.97` (`d = 16`).

---

## Reproduced with a caveat

**The `ε⁻²` exponent.** Fitted slope `−1.940 ± 0.026`, 95% CI `[−1.991, −1.890]`. The
scaling is clearly recovered, but at 40 seeds the interval is tight enough to *exclude*
`−2`, where the paper's wider `−1.92 ± 0.07` includes it. The residual is the bias the
paper itself names: `n*` is the first point of a discrete logarithmic grid at which the
dense model wins, which systematically overshoots, and the coupling term pushes the
large-`ε` crossovers further right. Reporting the fitted slope as consistent with `−2`
requires either acknowledging the grid bias or correcting for it, not simply a wide
enough error bar.

**The `d(d−1)` ratio.** Measured `n*` at `ε = 0.2`: 412 (`d = 8`), 899 (`d = 12`), 1612
(`d = 16`). Ratio `d = 16` over `d = 8` is **3.91** against the predicted 4.29 — closer
than the paper's reported 3.2 ± 0.3, in the same direction (measured below predicted,
consistent with the coupling term mattering more at the smaller `d = 8` crossover).

---

## Contradicted

**Coupling coefficient (Proposition 3).** Fitting `c_k` in
`E risk = δ_k² + σ²p_k/n + c_k δ_k²/n` at `ε = 0.8` with 400 seeds:

| k | `p_k` (paper's `c_k`) | measured `c_k` | `(d+1)/k` | measured ÷ `(d+1)/k` |
|---|---|---|---|---|
| 2 | 72 | 7.24 ± 0.23 | 6.50 | 1.11 |
| 3 | 48 | 4.82 ± 0.14 | 4.33 | 1.11 |
| 4 | 36 | 3.48 ± 0.12 | 3.25 | 1.07 |
| 6 | 24 | 2.27 ± 0.08 | 2.17 | 1.05 |
| 12 | 12 | 1.18 ± 0.05 | 1.08 | 1.09 |

The measured coefficient is about a tenth of `p_k` and within 5–11% of `(d+1)/k`, the
value a second-order expansion gives from `E[Δ²] = n(d+1)I` for the centred Wishart
fluctuation `Δ = Σ̂ − nI`. `tr(Π_k)` is indeed `p_k`, but the coupling term is not that
trace. This strengthens rather than weakens Theorem 5: the correction stays proportional
to `(1 − u)u`, stays concave, and is smaller than stated.

**Equation (7).** Against an ensemble of resampled graded targets, the piecewise profile
is off by up to 46%; the exact lattice-wide profile is within 2.5%. See the README.

**Proposition 8.** In the regular representation, `p_H = D²/|H|` holds for every subgroup
of all six groups tested, `γ_H` is affine in `1/|H|` for all of them, and no interior
optimum appears in any of 360 `(ε, n)` cells — including for the non-abelian `S₃` and
`D₄`, which have two-dimensional irreps. The printed `p_H = Σ m_λ² dim(λ)²` disagrees
with the true commutant dimension in 13 of the audited subgroups. Affineness does fail
for non-regular representations: in the natural representation of `S₄` on `R⁴`, `A₄` is
strictly optimal in 90 of 360 cells.

**Table 5's `d = 8` prediction.** 350, not the printed 280. Of 21 printed closed-form
constants audited, 18 check out; the three that do not are this one and the two ends of
the graded window quoted in Section 3.

**"Convex quadratic correction".** Concave in every cell checked.

---

## Validation tally

16 PASS, 4 PARTIAL, 6 FAIL out of 26 checks. All six FAILs are the discrepancies above;
none touches the central result, which is that under a generic defect the subgroup lattice
collapses to its endpoints and reopens when the defect is structured. Both halves of that
reproduce cleanly.
