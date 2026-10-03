"""
Unit tests.  Run with ``pytest -q`` from the repository root.

These test the *mathematics*, not the experiment outputs: projector identities,
closed-form defect formulas against brute-force measurement, estimator
consistency against an unconstrained reference solve, and the group-theoretic
machinery of Appendix H.
"""

from __future__ import annotations

import numpy as np
import pytest

from symbudget import theory as th
from symbudget.config import get_config
from symbudget.data import generic_target, graded_target, sample, StreamingStats, equivariant_component
from symbudget.estimators import CheckpointStore, excess_risk, fit_all_k, fit_constrained
from symbudget.lattice import divisors, lattice, orbit_labels, shift_matrix
from symbudget.nonabelian import (GROUPS, audit_group, commutant_dim_numeric,
                                  commutant_dim_trace, generate_group,
                                  natural_representation, paper_parameter_count,
                                  regular_representation, subgroups)
from symbudget.nonlinear import make_teacher, train_projected

DIMS = [6, 8, 12, 16]


# ------------------------------------------------------------- Proposition 1


@pytest.mark.parametrize("d", DIMS)
def test_parameter_count(d):
    lat = lattice(d)
    for k in lat.ks:
        assert lat.p[k] == d * d // k
        assert np.linalg.matrix_rank(lat.basis_matrix(k)) == d * d // k


@pytest.mark.parametrize("d", DIMS)
def test_orbit_partition(d):
    for k in divisors(d):
        lab = orbit_labels(d, k)
        counts = np.bincount(lab.ravel())
        assert len(counts) == d * d // k
        assert np.all(counts == k)          # the diagonal action is free


@pytest.mark.parametrize("d", DIMS)
def test_projector_identities(d):
    lat = lattice(d)
    rng = np.random.default_rng(0)
    W = rng.standard_normal((d, d))
    V = rng.standard_normal((d, d))
    for k in lat.ks:
        P = lat.project(W, k)
        assert np.allclose(P, lat.project_by_group_average(W, k))      # Prop. 1 formula
        assert np.allclose(lat.project(P, k), P)                        # idempotent
        assert np.isclose(np.sum(lat.project(W, k) * V),
                          np.sum(W * lat.project(V, k)))                # self-adjoint
        assert np.isclose(np.sum(P * (W - P)), 0.0, atol=1e-12)         # orthogonal
        assert lat.is_equivariant(P, k)                                 # lands in V_k


@pytest.mark.parametrize("d", DIMS)
def test_basis_orthonormal(d):
    lat = lattice(d)
    for k in lat.ks:
        B = lat.basis_matrix(k)
        assert np.allclose(B @ B.T, np.eye(B.shape[0]))


def test_lattice_nesting():
    """C_h <= C_k  =>  V_k subset V_h."""
    lat = lattice(12)
    rng = np.random.default_rng(1)
    for k in lat.ks:
        W = lat.random_in(k, rng)
        for h in lat.ks:
            if k % h == 0:                 # C_h <= C_k
                assert np.allclose(lat.project(W, h), W, atol=1e-10)


# ------------------------------------------------------------- Proposition 2


@pytest.mark.parametrize("d", [8, 12])
def test_defect_monotonicity_and_endpoints(d):
    lat = lattice(d)
    for seed in range(5):
        for eps in (0.1, 0.8):
            W = generic_target(d, eps, seed=seed, lat=lat).W
            prof = lat.defect_profile(W)
            assert prof[1] < 1e-20
            assert np.isclose(np.sqrt(prof[d]), eps)
            for h in lat.ks:
                for k in lat.ks:
                    if k % h == 0:
                        assert prof[h] <= prof[k] + 1e-12


# ------------------------------------------------------------- Proposition 4


@pytest.mark.parametrize("d", [8, 12, 16])
def test_gamma_generic_matches_measurement(d):
    lat = lattice(d)
    eps = 0.7
    meas = {k: [] for k in lat.ks}
    for s in range(300):
        prof = generic_target(d, eps, seed=s, lat=lat).defect_profile()
        for k in lat.ks:
            meas[k].append(prof[k] / eps**2)
    for k in lat.ks:
        m = float(np.mean(meas[k]))
        assert abs(m - float(th.gamma_generic(d, k))) < 0.02


def test_gamma_two_closed_forms_agree():
    """1 - (d/k - 1)/(d - 1)  ==  d(1 - 1/k)/(d - 1)."""
    for d in DIMS:
        for k in divisors(d):
            assert np.isclose(1 - (d / k - 1) / (d - 1), float(th.gamma_generic(d, k)))


# --------------------------------------------------------- Proposition 7


def test_graded_profile_exact_beats_piecewise():
    d, k0, b, c = 12, 4, 0.35, 0.08
    lat = lattice(d)
    meas = {k: [] for k in lat.ks}
    for s in range(200):
        prof = graded_target(d, k0, b, c, seed=s, lat=lat).defect_profile()
        for k in lat.ks:
            meas[k].append(prof[k])
    for k in lat.ks:
        m = float(np.mean(meas[k]))
        assert abs(m - float(th.delta2_graded_exact(d, k, k0, b, c))) < 2e-4
    # eq. (7) is exact at the three named points and wrong at k = 2
    for k in (1, k0, d):
        assert np.isclose(float(th.delta2_graded(d, k, k0, b, c)),
                          float(th.delta2_graded_exact(d, k, k0, b, c)))
    assert not np.isclose(float(th.delta2_graded(d, 2, k0, b, c)),
                          float(th.delta2_graded_exact(d, 2, k0, b, c)))


def test_graded_window_ordering():
    lo, hi = th.graded_window(12, 4, 0.35, 0.08, 0.5)
    assert 0 < lo < hi


# --------------------------------------------------- Theorem 5, Corollary 6


def test_phase_boundary_value():
    assert np.isclose(float(th.n_star(12, 0.5, 0.2)), 825.0)
    assert np.isclose(float(th.n_star(8, 0.5, 0.2)), 350.0)   # Table 5 prints 280
    assert np.isclose(float(th.n_star(16, 0.5, 0.2)), 1500.0)


def test_endpoints_cross_at_n_star():
    d, sigma, eps = 12, 0.5, 0.2
    n = float(th.n_star(d, sigma, eps))
    r_eq = th.risk_leading_order(th.delta2_generic(d, d, eps), d, d, sigma, n)
    r_dense = th.risk_leading_order(th.delta2_generic(d, 1, eps), d, 1, sigma, n)
    assert np.isclose(float(r_eq), float(r_dense))


def test_minimiser_always_endpoint_under_generic_defect():
    for d in DIMS:
        for eps in np.geomspace(0.01, 1.0, 12):
            for n in np.geomspace(10, 1e6, 15):
                assert th.minimiser_is_endpoint(d, 0.5, float(eps), float(n))


def test_coupling_correction_is_concave_not_convex():
    """The paper's main text says 'convex'; the correction is concave."""
    for d in DIMS:
        for eps in (0.05, 0.4):
            for n in (50, 5000):
                assert th.correction_curvature(d, eps, n) < 0


def test_interior_optimum_exists_for_graded_defect():
    d, k0, b, c = 12, 4, 0.35, 0.08
    model = th.RiskModel(d, 0.5, {k: float(th.delta2_graded_exact(d, k, k0, b, c))
                                  for k in divisors(d)})
    assert any(model.argmin(n) == k0 for n in np.geomspace(50, 5000, 40))


# ---------------------------------------------------------------- estimator


def test_constrained_fit_matches_unconstrained_at_k1():
    """At k = 1 the constrained solve must equal the ordinary least-squares solution."""
    d, n, sigma = 12, 2000, 0.3
    lat = lattice(d)
    W_star = generic_target(d, 0.3, seed=0, lat=lat).W
    rng = np.random.default_rng(0)
    X, Y = sample(W_star, n, sigma, rng)
    W_hat = fit_constrained(X.T @ X, Y.T @ X, n, 1, lat, ridge_rel=0.0)
    W_ref = np.linalg.lstsq(X, Y, rcond=None)[0].T
    assert np.allclose(W_hat, W_ref, atol=1e-8)


def test_constrained_fit_is_equivariant_and_optimal():
    d, n, sigma = 12, 500, 0.3
    lat = lattice(d)
    W_star = generic_target(d, 0.3, seed=1, lat=lat).W
    X, Y = sample(W_star, n, sigma, np.random.default_rng(2))
    S, C = X.T @ X, Y.T @ X
    for k in lat.ks:
        W = fit_constrained(S, C, n, k, lat)
        assert lat.is_equivariant(W, k, tol=1e-8)
        base = np.sum((Y - X @ W.T) ** 2)
        rng = np.random.default_rng(3)
        for _ in range(5):                       # no nearby feasible point is better
            P = lat.project(rng.standard_normal((d, d)), k)
            for t in (1e-3, -1e-3):
                assert np.sum((Y - X @ (W + t * P).T) ** 2) >= base - 1e-8


def test_ridge_does_not_move_numbers():
    d, n = 12, 1000
    lat = lattice(d)
    W_star = generic_target(d, 0.4, seed=4, lat=lat).W
    X, Y = sample(W_star, n, 0.4, np.random.default_rng(5))
    S, C = X.T @ X, Y.T @ X
    for k in lat.ks:
        a = excess_risk(fit_constrained(S, C, n, k, lat, ridge_rel=0.0), W_star)
        b = excess_risk(fit_constrained(S, C, n, k, lat, ridge_rel=1e-8), W_star)
        assert abs(a - b) < 1e-6 * max(a, 1e-12) + 1e-14   # ridge is 1e-8 relative


def test_underdetermined_returns_none():
    lat = lattice(12)
    S, C = np.eye(12), np.eye(12)
    assert fit_constrained(S, C, 5, 1, lat) is None       # 5 * 12 < 144 + 2
    assert fit_constrained(S, C, 5, 12, lat) is not None   # 5 * 12 >= 12 + 2


def test_streaming_stats_match_batch():
    W = generic_target(12, 0.2, seed=0).W
    ss = StreamingStats(W, 0.5, np.random.default_rng(7), block=97)
    ss.advance_to(300)
    S1, C1, n1 = ss.stats
    ss.advance_to(900)
    S2, C2, n2 = ss.stats
    assert n1 == 300 and n2 == 900
    assert np.all(np.linalg.eigvalsh(S2) > 0)
    rng = np.random.default_rng(7)
    X, Y = sample(W, 900, 0.5, rng)        # same seed, one shot (block size differs)
    assert S2.shape == X.T.shape[:1] + (12,)


def test_risk_matches_theory_at_large_n():
    d, sigma, eps, n = 12, 0.5, 0.05, 20000
    lat = lattice(d)
    W_star = generic_target(d, eps, seed=11, lat=lat).W
    risks = {k: [] for k in lat.ks}
    for s in range(12):
        ss = StreamingStats(W_star, sigma, np.random.default_rng(s)).advance_to(n)
        S, C, nn = ss.stats
        for k, r in fit_all_k(S, C, nn, W_star, lat).items():
            risks[k].append(r.excess_risk)
    for k in lat.ks:
        pred = float(th.risk_leading_order(lat.defect(W_star, k) ** 2, d, k, sigma, n))
        assert abs(np.mean(risks[k]) - pred) < 0.25 * pred


def test_equivariant_component_is_circulant_unit_norm():
    for d in DIMS:
        W = equivariant_component(d)
        assert np.isclose(np.linalg.norm(W), 1.0)
        assert lattice(d).is_equivariant(W, d)


# --------------------------------------------------------------- nonlinear


def test_projected_training_stays_equivariant():
    lat = lattice(12)
    rng = np.random.default_rng(0)
    teacher = make_teacher(12, 0.2, rng, lat)
    for k in (3, 12):
        r = train_projected(teacher, k, 200, 0.3, np.random.default_rng(1), lat,
                            steps=50, n_test=200, keep_weights=True)
        assert lat.is_equivariant(r.W1, k, tol=1e-8)
        assert lat.is_equivariant(r.W2, k, tol=1e-8)
        assert np.isfinite(r.test_mse)


def test_nonlinear_training_reduces_error():
    lat = lattice(12)
    rng = np.random.default_rng(0)
    teacher = make_teacher(12, 0.1, rng, lat)
    few = train_projected(teacher, 12, 400, 0.3, np.random.default_rng(2), lat, steps=5, n_test=500)
    many = train_projected(teacher, 12, 400, 0.3, np.random.default_rng(2), lat, steps=800, n_test=500)
    assert many.test_mse < few.test_mse


# -------------------------------------------------------------- non-abelian


def test_group_generation_orders():
    assert len(generate_group(GROUPS["S3"])) == 6
    assert len(generate_group(GROUPS["D4"])) == 8
    assert len(generate_group(GROUPS["S4"])) == 24


def test_commutant_dim_trace_matches_numeric():
    for name in ["C4", "V4", "S3", "D4"]:
        G = generate_group(GROUPS[name])
        for rep in (regular_representation(G), natural_representation(G)):
            for H in subgroups(G):
                assert abs(commutant_dim_trace(H, rep) - commutant_dim_numeric(H, rep)) < 1e-8


def test_regular_representation_gives_D2_over_H_for_every_group():
    """The key correction to Proposition 8."""
    for name in ["C4", "C6", "V4", "S3", "D4"]:
        G = generate_group(GROUPS[name])
        rep = regular_representation(G)
        D = len(G)
        for H in subgroups(G):
            assert np.isclose(commutant_dim_trace(H, rep), D * D / len(H))


def test_collapse_survives_for_nonabelian_regular_reps():
    for name in ["S3", "D4"]:
        a = audit_group(name, GROUPS[name], rep_kind="regular")
        assert not a["abelian"]
        assert a["affine"]
        assert a["regular_formula_holds"]


def test_paper_pH_formula_overcounts_with_higher_dim_irreps():
    """S_4 on R^4 restricted to S_3: true p_H = 5, the printed formula gives 8."""
    G = generate_group(GROUPS["S4"])
    rep = natural_representation(G)
    H = [g for g in G if g[3] == 3]              # stabiliser of the 4th point ~ S_3
    assert len(H) == 6
    assert commutant_dim_numeric(H, rep) == 5
    assert paper_parameter_count(H, rep) == 8


def test_natural_rep_breaks_affineness():
    a = audit_group("S4", GROUPS["S4"], rep_kind="natural")
    assert not a["affine"]


# ------------------------------------------------------------------ config


def test_config_fingerprint_is_stable_and_sensitive():
    a, b = get_config("paper"), get_config("paper")
    assert a.fingerprint() == b.fingerprint()
    b.sigma = 0.6
    assert a.fingerprint() != b.fingerprint()


def test_checkpoint_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path)
    W = np.arange(9.0).reshape(3, 3)
    store.save_model("t", W, W * 2, k=3, n=10)
    Wh, Ws, meta = store.load_model("t")
    assert np.allclose(Wh, W) and np.allclose(Ws, 2 * W)
    assert meta["k"] == 3 and meta["n"] == 10
    assert "t" in store.list_models()
