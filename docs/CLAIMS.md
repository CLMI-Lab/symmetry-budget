# Claim → code → output map

Every quantitative statement in the paper, where it is implemented, where the number it
produces lands, and which check in `results/validation.csv` adjudicates it.

The verdicts themselves are regenerated on every run; this file is the static index.

| # | Paper location | Claim | Implemented in | Output | Check id |
|---|---|---|---|---|---|
| 1 | Proposition 1 | `dim V_k = d²/k` | `lattice.orbit_labels`, `lattice.SubgroupLattice.basis` | `theory_checks_prop1.csv` | `P1-dim` |
| 2 | Proposition 1 | `Π_k(W) = (1/k) Σ_m S^{−md/k} W S^{md/k}` is the orthogonal projector onto `V_k` | `lattice.project`, `lattice.project_by_group_average` | `theory_checks_prop1.csv` | `P1-proj` |
| 3 | Proposition 2 | `C_h ≤ C_k ⟹ δ_h ≤ δ_k` | `lattice.defect_profile` | `theory_checks_prop2.csv` | `P2-mono` |
| 4 | Proposition 2 | `δ₁ = 0`, `δ_d = ε` | `data.generic_target`, `lattice.defect` | `theory_checks_endpoints.csv` | `P2-ends` |
| 5 | Proposition 3 | leading-order risk `δ_k² + σ²p_k/n` | `theory.risk_leading_order` | `phase_diagram_d12.csv` | `P3-variance` |
| 6 | Proposition 3, eq. (3) | coupling term `δ_k² tr(Π_k)/n` with `tr(Π_k) = p_k` | `theory.risk_finite_sample`, `experiments.run_variance_constant` | `coupling_fit.csv` | `P3-coupling` |
| 7 | Appendix C | `ρ_k ∈ [0.97, 1.04]` at `n = 4000` | `experiments.run_variance_constant` | `table4_variance_constant.csv` | `P3-variance` |
| 8 | Proposition 4 | `γ_k = d(1 − 1/k)/(d − 1)`, within 1% | `theory.gamma_generic`, `experiments.run_generic_defect` | `table1_generic_defect.csv` | `P4-gamma` |
| 9 | Theorem 5 | minimiser is always an endpoint under a generic defect | `theory.minimiser_is_endpoint`, `experiments.run_phase_diagram` | `phase_cells_d12.csv` | `T5-endpoint` |
| 10 | Theorem 5 / Sec. 4.1 | predicted `k*` agrees with measured on every cell | `experiments._phase_diagram` | `phase_cells_d12.csv` | `T5-cells` |
| 11 | Theorem 5 / Sec. 3 | the correction is a "convex quadratic" | `theory.correction_curvature` | `theory_checks_curvature.csv` | `T5-curv` |
| 12 | Corollary 6 | `n* = σ²d(d−1)/ε²` | `theory.n_star` | `table2_crossover.csv` | `C6-constant` |
| 13 | Sec. 4.2 | fitted exponent `−1.92 ± 0.07` against `−2` | `experiments._fit_exponent` | `table2_crossover.csv` | `C6-exponent` |
| 14 | Appendix F | per-seed IQR below the grid spacing | `experiments._crossover_for_d` | `table2_crossover.csv` | `C6-iqr` |
| 15 | Sec. 4.3 | `k*` agrees on 32/32 at `d = 8`, 48/48 at `d = 16` | `experiments.run_dimension` | `phase_cells_d8.csv`, `phase_cells_d16.csv` | `D-cells-8`, `D-cells-16` |
| 16 | Sec. 4.3 | exponents `−1.94`, `−1.91` within the CI of `−2` | `experiments.run_dimension` | `crossover_d8.csv`, `crossover_d16.csv` | `D-exp-8`, `D-exp-16` |
| 17 | Sec. 4.3 / Table 5 | the `d(d−1)` ratio | `experiments.run_dimension` | `table5_dimension.csv` | `D-ratio` |
| 18 | Proposition 7, eq. (7) | the graded defect profile, to within 3% | `theory.delta2_graded`, `theory.delta2_graded_exact` | `graded_defect_profile.csv` | `P7-profile` |
| 19 | Sec. 4.4 | the ladder `12 → 4 → 1` | `experiments.run_graded` | `graded_ladder.csv` | `P7-ladder` |
| 20 | Proposition 7, eq. (8) | the window `80 ≲ n ≲ 2500` | `theory.graded_window` | `graded_ladder.csv` | `P7-window` |
| 21 | Sec. 4.5 / Table 3 | the crossover survives nonlinearity and moves with `ε` | `nonlinear.train_projected` | `table3_nonlinear.csv` | `NL-crossover` |
| 22 | Proposition 8 | collapse holds iff every irrep of `H` is one-dimensional | `nonabelian.audit_group` | `nonabelian_audit.csv` | `P8-regular` |
| 23 | Proposition 8 | `p_H = Σ_λ m_λ² dim(λ)²` | `nonabelian.paper_parameter_count`, `nonabelian.commutant_dim_numeric` | `nonabelian_audit.csv` | `P8-pH` |
| 24 | Appendix H | a setting where the collapse genuinely fails | `nonabelian.interior_optimum_scan` | `nonabelian_audit.csv` | `P8-counterexample` |
| 25 | Tables 1, 2, 5; Sec. 3 | the printed closed-form constants | `experiments.run_theory_checks` | `paper_constants_audit.csv` | `ARITH` |
| 26 | Table 6 | which quantities are closed-form vs sampled | `scripts/make_provenance.py` | `table6_provenance.csv` | — |
| 27 | Appendix B / F | per-architecture cost `O(nd² + d³)` | `estimators.fit_constrained` | `results.json` timings | — |
| 28 | Appendix B | the `1e-8` relative ridge changes nothing reported | `estimators.fit_constrained` | `tests::test_ridge_does_not_move_numbers` | — |
| 29 | Appendix E | the two expectations (data, defect direction) are kept separate | `experiments._risk_curves` vs `run_generic_defect` | `table1_generic_defect.csv` | `P4-gamma` |

## Verdict legend

- **PASS** — this implementation reproduces the claim. It does *not* mean the paper's exact
  digits were reproduced; seeds and defect draws differ.
- **PARTIAL** — reproduced in substance but not in the stated number, or reproduced only
  under a coarser preset.
- **FAIL** — the implementation contradicts the claim. Each FAIL carries a note explaining
  the discrepancy; see the "five discrepancies" section of the README.

## Things the paper claims that this implementation deliberately does **not** test

- *"We do not claim that ImageNet architectures obey ε⁻²."* No real-data experiment is
  implemented, consistent with Section 6: instantiating `δ_k` for CIFAR-10 requires
  estimating the defect from data, which the paper itself calls an open problem.
- The heteroscedastic-noise remark in Limitations (ii) is a statement about a model the
  paper does not analyse, so there is nothing to check.
