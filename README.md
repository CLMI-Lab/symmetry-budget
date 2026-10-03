# The Symmetry Budget — reference implementation

A complete, self-contained implementation of **"The Symmetry Budget: How Much Group
Structure Should a Network Architecture Assume?"** (Mohammad, Hasan, Shammo, Siddiki,
Habib; SGNR / JMLR proceedings track, 2026).

Everything in the paper is implemented: the closed-form theory (Propositions 1–8,
Theorem 5, Corollary 6), the constrained-least-squares estimator, all five experiments,
the figures, the tables, model checkpoints, and a **claim-by-claim validation harness**
that turns every quantitative statement in the paper into a machine-checked predicate.

CPU only. No GPU, no MPI, no multithreading. The full protocol runs in minutes.

---

## Quick start

```bash
pip install -r requirements.txt

python run_all.py                    # full protocol (Appendix B)
python run_all.py --preset quick     # coarser grids, ~1 minute
python run_all.py --preset smoke     # minimal, for CI
python -m pytest tests -q            # 47 unit tests
```

Selected stages and resumption:

```bash
python run_all.py --stages graded,nonlinear
python run_all.py --resume           # reuse cached stage results
python run_all.py --no-figures
```

---

## What gets produced

```
results/
├── csv/                       every table, in paper format and long format
│   ├── table1_generic_defect.csv        Table 1   γ_k closed form vs measurement
│   ├── table2_crossover.csv             Table 2   n*(ε) with IQR and ratio
│   ├── table3_nonlinear.csv             Table 3   two-layer tanh control
│   ├── table4_variance_constant.csv     Table 4   ρ_k across defect sizes
│   ├── table5_dimension.csv             Table 5   d-dependence of n*
│   ├── phase_diagram_d{8,12,16}.csv     long format: (d, ε, n, k) → risk
│   ├── phase_cells_d{8,12,16}.csv       per-cell measured vs predicted k*
│   ├── crossover_d{8,12,16}.csv         crossover per dimension
│   ├── graded_defect_profile.csv        measured vs eq. (7) vs exact profile
│   ├── graded_ladder.csv                the k* ladder over n
│   ├── coupling_fit.csv                 direct measurement of the coupling term
│   ├── nonabelian_audit.csv             Proposition 8 over groups and reps
│   ├── paper_constants_audit.csv        printed constants vs their formulas
│   └── theory_checks_*.csv              projector and monotonicity checks
├── figures/                   Figures 1–5, each as PDF and PNG
├── checkpoints/
│   ├── models/                fitted Ŵ_k (+ W*) with JSON metadata, reloadable
│   └── stages/                stage caches for --resume
├── results.json               every headline number, config, environment, timings
├── VALIDATION.md              claim-by-claim verdicts with notes
├── validation.csv             the same, machine-readable
└── manifest.json              file inventory with sizes and sha256 prefixes
```

Reload a checkpoint:

```python
from symbudget.estimators import CheckpointStore
store = CheckpointStore("results/checkpoints")
W_hat, W_star, meta = store.load_model("phase_d12_eps0.4_n4000_k12")
meta  # {'d': 12, 'k': 12, 'n': 4000, 'eps': 0.4, 'p': 12, 'excess_risk': ..., 'delta2': ...}
```

---

## Package layout

| module | contents |
|---|---|
| `symbudget/lattice.py` | divisor lattice of `C_d`, orbit basis, projections `Π_k` (Prop. 1–2) |
| `symbudget/theory.py` | `γ_k`, defects, risk, `n*`, endpoint test, graded window (Prop. 3–4, 7; Thm. 5; Cor. 6) |
| `symbudget/data.py` | the model of eq. (2), generic and graded targets, streaming sufficient statistics |
| `symbudget/estimators.py` | constrained least squares, excess risk, checkpoint store |
| `symbudget/nonlinear.py` | two-layer tanh control with projected gradient descent |
| `symbudget/nonabelian.py` | finite groups, representations, commutant dimensions, Prop. 8 audit |
| `symbudget/config.py` | presets `paper` / `quick` / `smoke`, with a config fingerprint |
| `symbudget/experiments.py` | the nine experiment stages |
| `symbudget/figures.py` | Figures 1–3 plus two diagnostics |
| `symbudget/validate.py` | the claim-checking harness |

---

## Algorithms

**Algorithm 1 — orbit basis.** For `k | d`, `s = d/k`, the diagonal action
`(i, j) ↦ (i+s, j+s)` on `Z_d × Z_d` is free, so its orbits all have size `k` and there
are `d²/k` of them. `V_k` is exactly the set of matrices constant along orbits, and the
normalised orbit indicators form an orthonormal basis. `Π_k` is the orbit average; the
implementation also provides the literal group-average form of Proposition 1 and the two
are asserted equal in the test-suite.

**Algorithm 2 — constrained least squares via sufficient statistics.** Writing
`W = Σ_a θ_a B_a`, the normal equations are `G θ = r` with
`G_{ba} = ⟨B_b, B_a Σ̂⟩`, `r_b = ⟨B_b, C⟩`, `Σ̂ = XᵀX`, `C = YᵀX`. Both statistics are
`d × d`, so the final solve is independent of `n`; forming them is `O(n d²)` and the solve
`O(p_k³ + p_k d³)`. Per-architecture cost `O(n d² + d³)` — the figure Appendix F corrects
to. A relative ridge of `1e-8 · tr(G)/p_k` is added; `tests/test_symbudget.py` asserts it
changes nothing at the quoted precision. Configurations with `n d < p_k + 2` return NaN.

**Algorithm 3 — architecture selection.** The minimiser over the lattice of the
finite-sample risk (3), with ties broken towards the more constrained model.

**Algorithm 4 — projected gradient descent.** Two-layer tanh student against a matched
teacher, full-batch GD, both weight matrices initialised inside `V_k` and every gradient
projected onto `V_k` before the update, so the iterates never leave the constraint set.

**Nested-grid trick.** `StreamingStats` accumulates `Σ̂` and `C` in blocks, so a grid
`n₁ < … < n_m` costs one pass over `n_m` rather than `m` passes. This is what makes the
60-point logarithmic crossover grid at 40 seeds run in about 14 seconds.

---

## Reproduced

| claim | status |
|---|---|
| Prop. 1 — `dim V_k = d²/k`, `Π_k` the orthogonal projector | exact; rank matches at `d ∈ {8,12,16}`, projector identities to `2e-16` |
| Prop. 2 — monotonicity, `δ₁ = 0`, `δ_d = ε` | exact on all lattice pairs |
| Prop. 4 — `γ_k = d(1 − 1/k)/(d − 1)` | reproduced within sampling error |
| Thm. 5 — the minimiser is always an endpoint under a generic defect | reproduced; no interior subgroup ever selected |
| Cor. 6 — `n* = σ²d(d−1)/ε²`, exponent `−2` | reproduced; fitted slope's 95% CI contains `−2` |
| Prop. 7 — the lattice reopens, ladder `12 → 4 → 1` | reproduced |
| Sec. 4.5 — the crossover survives nonlinearity | reproduced |
| Appendix C — graded defect profile `0, 0.0043, 0.1268, 0.0064, 0.1282, 0.1289` | reproduced to the printed digits |

See `results/VALIDATION.md` for the full per-claim table with measured values.

---

## Five discrepancies the implementation turns up

These are reported because the implementation checks them, not because they damage the
paper's main result — the central finding (all-or-nothing collapse under a generic
defect, reopening under a structured one) survives all five.

**1. Theorem 5's "convex" should be "concave".** With `u = 1/k` the coupling correction is
proportional to `(1 − u)u`, whose second derivative is negative. Section 3 calls it a
"convex quadratic correction" and then argues that *"a convex function on a finite set
attains its minimum at an extreme point"* — false for convex functions, true for concave
ones. Appendix A and Appendix D of the same paper say "concave" and give the correct
argument. The endpoint conclusion is right; the main-text wording is a typo.
Checked by `theory.correction_curvature` and `test_coupling_correction_is_concave_not_convex`.

**2. Eq. (7) is not exact across the whole lattice.** The three-case profile is exact only
at `k ∈ {1, k₀, d}` and for `k ∤ k₀`. For an intermediate `k | k₀` with `k < k₀`, the
architecture recovers `u_{k₀}` entirely but only *part* of `u₁`, so the true defect is
strictly below `c²`. At `k = 2`, `k₀ = 4`, `d = 12`, `(b, c) = (0.35, 0.08)` eq. (7) gives
0.0064 while measurement gives 0.0043 — and the paper's own Appendix C reports 0.0043, so
the "matches to within 3%" claim does not hold for eq. (7) as printed (the error is 31%
at that point). `theory.delta2_graded_exact` gives the exact lattice-wide profile,

```
δ²_k = b² (p_{k₀} − p_L)/(p_{k₀} − p_d) + c² (p₁ − p_k − p_{k₀} + p_L)/(p₁ − p_{k₀}),
L = lcm(k, k₀),
```

using `V_k ∩ V_{k₀} = V_{lcm(k,k₀)}`; it matches measurement at every `k` to five decimals.
The ladder and the window are unaffected.

**3. The coupling coefficient appears to be about an order of magnitude too large.**
Proposition 3 asserts the coupling term is `δ_k² tr(Π_k)/n` with `tr(Π_k) = p_k = d²/k`.
`tr(Π_k)` is indeed `p_k`, but the coupling term is not that trace. A second-order
expansion gives `c_k ≈ (d+1)/k`: the fluctuation `Δ = Σ̂ − nI` of the centred Wishart
satisfies `E[Δ²] = n(d+1)I`, and `‖Π_k(W⊥ Δ)‖²/n² ≈ δ²(d+1)/(kn)`. The `coupling_fit`
stage measures `c_k` directly at high seed count and the measurement sits near the smaller
value. This makes the finite-sample correction *smaller* than stated, so Theorem 5 is if
anything on firmer ground — the correction is still proportional to `(1 − u)u` and still
concave.

**4. Proposition 8 states the wrong criterion.** In the **regular** representation every
non-identity element has zero trace, so

```
p_H = (1/|H|) Σ_h tr(ρ(h))² = |G|²/|H|
```

for *every* subgroup of *every* finite group, giving `γ_H = |G|(1 − 1/|H|)/(|G| − 1)` —
identical to the cyclic formula. The audit confirms this for `S₃` and `D₄`, both
non-abelian with two-dimensional irreps: affine, and no interior optimum in any of 360
`(ε, n)` cells. Since `C_d` acting on `R^d` by shifts *is* its regular representation, that
is the setting the paper generalises, so commutativity and irrep dimension are not what
governs the collapse. Separately, the printed `p_H = Σ_λ m_λ² dim(λ)²` overcounts: the
commutant of `⊕ m_λ λ` has dimension `Σ m_λ²`. For `S₄` on `R⁴` restricted to `S₃`
(`2 × trivial + 1 × standard`) the true value is `2² + 1² = 5`, confirmed by direct rank
computation, while the printed formula gives `4·1 + 1·4 = 8`.

The paper's *qualitative* conclusion is nevertheless correct, for the right reason:
affineness is a property of the **representation**. In the natural permutation
representation of `S₄` on `R⁴`, `γ_H` is not affine in `1/|H|` and the subgroup of order 12
(`A₄`) is strictly optimal in 90 of 360 cells. The corrected statement: *the affine
collapse survives iff `p_H` is affine in `1/|H|`, which holds automatically in the regular
representation of any finite group and can fail otherwise.*

**5. Two printed constants.** Table 5 gives a predicted `n*` of 280 at `d = 8`, `ε = 0.2`,
`σ = 0.5`, but `σ²d(d−1)/ε² = 0.25·8·7/0.04 = 350`; the `d = 12` (825) and `d = 16` (1500)
entries of the same row are correct, so this is an isolated slip, and it moves the reported
`d = 8` ratio from 1.11 to about 0.89. Section 3 also states the predicted graded window as
`80 ≲ n ≲ 2500` while eq. (8) with `(b, c) = (0.35, 0.08)` evaluates to roughly
`[49, 4219]` — the measured window still sits inside the formula's interval, so this is a
reporting rather than a substantive problem. All of Table 2's predicted values check out.
See `results/csv/paper_constants_audit.csv`.

---

## Reproducibility

Every number is either a closed-form expression evaluated in `symbudget/theory.py` or a
sample statistic from an explicitly listed seed set; `results/csv/table6_provenance.csv`
records which. The configuration is hashed into `results.json`, so a stale checkpoint can
never be silently reused. Seeds are `0 … n_seeds-1` for the risk experiments and a
disjoint block for the defect-resampling and coupling stages, so the two kinds of
expectation (over training data, and over the defect direction) are never conflated — the
distinction Appendix E draws.

## Citation

```bibtex
@inproceedings{mohammad2026symmetrybudget,
  title     = {The Symmetry Budget: How Much Group Structure Should a Network
               Architecture Assume?},
  author    = {Mohammad, Noor Islam S. and Hasan, Mahmudul and Shammo, Md. Basim Al Zabir
               and Siddiki, Hasan and Habib, Jakaria},
  booktitle = {Symmetry and Geometry in Neural Representations},
  year      = {2026}
}
```

## License

MIT.
