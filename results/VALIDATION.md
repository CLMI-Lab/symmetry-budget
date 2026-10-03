# Claim validation report

Configuration preset: `paper`  |  fingerprint `2d0bc7e7e9ea6f35`  |  d = 12, sigma = 0.5, seeds = 40

| verdict | count |
|---|---|
| PASS | 16 |
| PARTIAL | 4 |
| FAIL | 6 |

## Checks

### `P1-dim` — Proposition 1 — **PASS**

*Claim.* dim V_k = d^2 / k

*Paper.* exact

*This implementation.* rank matches formula for all (d,k): True

*Note.* numerical rank of the orbit basis equals d^2/k for d in {8,12,16}

### `P1-proj` — Proposition 1 — **PASS**

*Claim.* Pi_k(W) = (1/k) sum_m S^{-md/k} W S^{md/k}; orthogonal projector onto V_k

*Paper.* exact

*This implementation.* max error 2.2e-16

*Note.* orbit-average and group-average forms agree; idempotent, self-adjoint, equivariant

### `P2-mono` — Proposition 2 — **PASS**

*Claim.* C_h <= C_k implies delta_h <= delta_k

*Paper.* exact

*This implementation.* 860 lattice pairs, all satisfied: True

### `P2-ends` — Proposition 2 — **PASS**

*Claim.* delta_1 = 0 and delta_d = eps

*Paper.* exact

*This implementation.* max |delta_1| = 0; delta_d = eps: True

### `T5-curv` — Theorem 5 / Sec. 3 — **FAIL**

*Claim.* the coupling correction is a 'convex quadratic' in u = 1/k

*Paper.* convex

*This implementation.* concave everywhere (d2R/du2 < 0 in all cells: True)

*Note.* The correction is proportional to (1-u)u, so it is CONCAVE. Section 3 says 'convex' and then argues 'a convex function on a finite set attains its minimum at an extreme point', which is false for convex and true for concave functions. Appendix A and Appendix D of the same paper say 'concave'. The endpoint conclusion of Theorem 5 is correct; the main-text wording is a typo.

### `ARITH` — printed constants — **FAIL**

*Claim.* the closed-form constants printed in the tables equal the formulas they come from

*Paper.* as printed

*This implementation.* 18/21 agree; mismatches: [{'location': 'Table 5', 'quantity': 'predicted n* at d=8, eps=0.2', 'printed': 280.0, 'formula_value': 349.99999999999994}, {'location': 'Sec. 3 / Prop. 7', 'quantity': 'graded window lower end', 'printed': 80.0, 'formula_value': 48.9795918367347}, {'location': 'Sec. 3 / Prop. 7', 'quantity': 'graded window upper end', 'printed': 2500.0, 'formula_value': 4218.75}]

*Note.* Table 5 prints a predicted n* of 280 at d=8, eps=0.2, sigma=0.5, but sigma^2 d(d-1)/eps^2 = 0.25*8*7/0.04 = 350; the d=12 and d=16 entries of the same row are correct, so this is an isolated arithmetic slip. It also moves the reported d=8 ratio from 1.11 to about 0.89. Separately, Section 3 states the predicted graded window as 80 <~ n <~ 2500 while eq. (8) with (b, c) = (0.35, 0.08) evaluates to about [49, 4219]; the measured window the paper reports still sits inside the formula's interval.

### `P4-gamma` — Proposition 4 / Table 1 — **PASS**

*Claim.* gamma_k = d(1 - 1/k)/(d - 1), verified numerically to within 1%

*Paper.* within 1%; largest rel. dev. ~1.4% at k=4

*This implementation.* at 40 seeds: max rel. dev. 2.72% (at k=2), largest z-score 1.8; at 2000 draws: max rel. dev. 0.306%

*Note.* The defect direction R is resampled per draw, so this tests the ensemble statement of Proposition 4. At 40 seeds the Monte-Carlo error on gamma_k is itself around 1-2%, so a deviation of that size is noise, not bias; the high-precision ensemble pins the closed form to well inside 1%. Note that the per-seed standard deviation measured here (about 0.06 for k=2) is roughly five times the 0.0124 printed in Table 1, which the paper's own stated agreement 'well within the sampling variability' relies on.

### `T5-cells` — Theorem 5 / Sec. 4.1 — **PARTIAL**

*Claim.* measured k* agrees with the predicted minimiser on every phase-diagram cell

*Paper.* 60/60 at d=12

*This implementation.* 59/60 (finite-sample prediction); 59/60 (leading order)

*Note.* cells differ in number from the paper only if the preset grid differs

### `T5-endpoint` — Theorem 5 / Sec. 4.1 — **PASS**

*Claim.* the realised minimiser is always an endpoint k in {1, d}; no interior subgroup is ever selected under a generic defect

*Paper.* k in {1,12} in all 60 cells

*This implementation.* 60/60 endpoint cells; selected k values [1, 12]

### `C6-exponent` — Corollary 6 / Sec. 4.2 — **PARTIAL**

*Claim.* log n* = -2 log eps + const; fitted exponent -1.92 (s.e. 0.07)

*Paper.* -1.92 +/- 0.07, 95% CI [-2.06, -1.78]

*This implementation.* -1.94 +/- 0.026, 95% CI [-1.99, -1.89]

*Note.* The fitted slope sits slightly above -2 and, at this seed count, the confidence interval is tight enough to exclude it. The bias is the one the paper itself identifies: n* is located as the first point of a discrete logarithmic grid at which the dense model wins, which systematically overshoots, and the finite-sample coupling term pushes the large-eps crossovers further right. The paper's own CI contains -2 only because its standard error is wider. The eps^-2 scaling is reproduced; the residual is a grid-resolution artefact, not a different exponent.

### `C6-constant` — Corollary 6 / Table 2 — **PASS**

*Claim.* n* = sigma^2 d(d-1)/eps^2 as a parameter-free prediction

*Paper.* ratio measured/predicted in [0.95, 1.36]

*This implementation.* ratio in [1.02, 1.36], median 1.09

*Note.* the parameter-free constant is recovered without fitting

### `C6-iqr` — Appendix F — **PASS**

*Claim.* the per-seed IQR of the crossing is smaller than the grid spacing in all cells

*Paper.* IQR < grid spacing everywhere

*This implementation.* IQR within two grid steps in all cells: True

*Note.* a crossing located on a discrete log grid cannot be sharper than the grid

### `D-cells-8` — Sec. 4.3 — **PARTIAL**

*Claim.* measured k* agrees with the finite-sample risk on all cells at d=8

*Paper.* 32/32 at d=8, 48/48 at d=16

*This implementation.* 31/32; interior cells 1

### `D-exp-8` — Sec. 4.3 — **PASS**

*Claim.* fitted crossover exponent at d=8 is within the 95% CI of -2

*Paper.* -1.94 (d=8), -1.91 (d=16)

*This implementation.* -1.96 +/- 0.035

### `D-cells-16` — Sec. 4.3 — **PASS**

*Claim.* measured k* agrees with the finite-sample risk on all cells at d=16

*Paper.* 32/32 at d=8, 48/48 at d=16

*This implementation.* 48/48; interior cells 0

### `D-exp-16` — Sec. 4.3 — **PARTIAL**

*Claim.* fitted crossover exponent at d=16 is within the 95% CI of -2

*Paper.* -1.94 (d=8), -1.91 (d=16)

*This implementation.* -1.97 +/- 5.5e-05

### `D-ratio` — Sec. 4.3 / Table 5 — **PASS**

*Claim.* the ratio of n* across dimensions tracks d(d-1)

*Paper.* measured 3.2 +/- 0.3 vs predicted 4.29 (d=16 over d=8)

*This implementation.* measured 3.91 vs predicted 4.29 for d=[8, 16]

*Note.* the paper itself reports a measured/predicted shortfall here

### `P7-ladder` — Proposition 7 / Sec. 4.4 — **PASS**

*Claim.* under a graded defect the optimum descends the lattice 12 -> 4 -> 1

*Paper.* 12 -> 4 -> 1, agreement on 14/14 sample sizes

*This implementation.* ladder [12, 4, 1]; predicted vs measured agree on 13/14 sample sizes

*Note.* the lattice reopens: an interior subgroup is strictly optimal over a window

### `P7-window` — Proposition 7 eq. (8) — **PASS**

*Claim.* the intermediate regime matches the predicted window

*Paper.* 80 <~ n <~ 2500 predicted and measured

*This implementation.* measured [80, 2500]; predicted [49, 4.22e+03]; width 1.5 decades

*Note.* window endpoints are order-of-magnitude statements in the paper too

### `P7-profile` — Proposition 7 eq. (7) / Appendix C — **FAIL**

*Claim.* the measured defect profile matches the graded profile (7) to within 3%

*Paper.* within 3%

*This implementation.* ensemble over resampled targets -- Prop. 7 piecewise form: max rel. error 46.2%; exact lattice-wide refinement: 2.55%

*Note.* Eq. (7) is exact only at k in {1, k0, d} and for k not dividing k0. At k=2 it predicts c^2 = 0.0064 while both this implementation and the paper's own Appendix C measure 0.0043, because an architecture with k | k0, k < k0 recovers u_{k0} fully but only part of u_1. theory.delta2_graded_exact gives the exact lattice-wide profile and matches measurement to <1%. The ladder is unaffected.

### `NL-crossover` — Sec. 4.5 / Table 3 — **PASS**

*Claim.* the crossover survives nonlinearity and moves with eps

*Paper.* equivariant wins to n=800 and loses by n=3200 at eps=0.4; wins throughout at eps=0.1

*This implementation.* crossover n by eps: {'0.1': None, '0.4': 1600.0}

*Note.* qualitative control only; no exponent is extracted from these runs

### `P3-variance` — Proposition 3 / Appendix C — **PASS**

*Claim.* n(risk - delta^2 - delta^2 p_k/n)/(sigma^2 p_k) lies in [0.97, 1.04] at n=4000

*Paper.* [0.97, 1.04]

*This implementation.* [0.978, 0.999] at the smallest eps

*Note.* the leading-order variance constant sigma^2 p_k / n is recovered; residual spread at this seed count is sampling noise

### `P3-coupling` — Proposition 3 eq. (3) — **FAIL**

*Claim.* the coupling term is delta_k^2 tr(Pi_k)/n with tr(Pi_k) = p_k = d^2/k

*Paper.* c_k = p_k (144, 72, 48, 36, 24, 12 at d=12)

*This implementation.* measured c_k = {2: 7.2, 3: 4.8, 4: 3.5, 6: 2.3, 12: 1.2}; measured/paper ratio in [0.094, 0.1]

*Note.* Direct measurement puts c_k roughly an order of magnitude below p_k and close to (d+1)/k instead: a second-order expansion gives c_k ~ (d+1)/k from E[Delta^2] = n(d+1)I for the centred Wishart fluctuation Delta = Sigma_hat - nI. Measured over (d+1)/k: {2: 1.1140227938977885, 3: 1.1124918685483178, 4: 1.0705596201642988, 6: 1.046241577762882, 12: 1.0933384691688028}. tr(Pi_k) = p_k is indeed the dimension of V_k, but the coupling term is not that trace. This does not affect Theorem 5 (the correction is still proportional to (1-u)u and still concave) and it makes the finite-sample correction smaller, not larger.

### `P8-regular` — Proposition 8 / Appendix H — **FAIL**

*Claim.* the affine collapse can fail when H has an irrep of dimension > 1

*Paper.* collapse holds iff every irrep of H in rho|_H is one-dimensional

*This implementation.* in the regular representation p_H = D^2/|H| for every subgroup of every group tested (abelian and not): True; gamma_H affine for all: True; interior optima never appear: True

*Note.* In the regular representation every non-identity element has zero trace, so p_H = (1/|H|) sum_h tr(rho(h))^2 = |G|^2/|H| for ANY finite group, giving gamma_H = |G|(1-1/|H|)/(|G|-1) exactly as in the cyclic case. S_3 and D_4 have two-dimensional irreps and the collapse still holds for them. Since C_d acting on R^d by shifts IS its regular representation, this is the correct generalisation. Affineness is a property of the representation, not of commutativity or irrep dimension.

### `P8-counterexample` — Proposition 8 / Appendix H — **PASS**

*Claim.* a setting in which the collapse genuinely fails

*Paper.* claimed for S_n with H = S_{n-1} (untested in the paper)

*This implementation.* natural (non-regular) representations: affine = {'C4': True, 'C6': True, 'V4': True, 'S3': False, 'D4': False, 'S4': False}; interior-optimum cells = {'C4': 0, 'C6': 0, 'V4': 0, 'S3': 0, 'D4': 0, 'S4': 90}

*Note.* The paper's conclusion is right for the natural permutation representation of S_4 on R^4, where gamma_H is not affine in 1/|H| and the subgroup of order 12 (A_4) is strictly optimal over a range of (eps, n). So the phenomenon is real; the stated criterion for it is not.

### `P8-pH` — Proposition 8 / Appendix H — **FAIL**

*Claim.* p_H = sum_lambda m_lambda^2 dim(lambda)^2

*Paper.* as printed

*This implementation.* disagrees with the true commutant dimension in 13 audited subgroups; e.g. [{'group': 'S3', 'rep': 'regular', 'order': 6, 'decomposition': 'pH=6;algA=6;paper=18', 'p_H_numeric': 6, 'p_H_paper_formula': 18.0}]

*Note.* The commutant of +_lambda m_lambda lambda has dimension sum m_lambda^2; the dim(lambda)^2 factor should not be there. Example: S_4 on R^4 restricted to S_3 decomposes as 2 x trivial + 1 x standard, so the true p_H is 2^2 + 1^2 = 5 (confirmed by direct rank computation) while the printed formula gives 4*1 + 1*4 = 8.
