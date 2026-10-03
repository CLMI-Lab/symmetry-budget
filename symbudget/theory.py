"""
Closed-form theory: Propositions 3, 4, 7, Theorem 5, Corollary 6.

Every function here is an exact evaluation of a formula in the paper.  Nothing
in this module touches data, so nothing in it depends on a random seed; that
separation is what Table 6 of the paper records and what
``symbudget.validate`` checks.

Sign convention note (important, and a correction to the paper's main text)
--------------------------------------------------------------------------
With u = 1/k the finite-sample risk of eq. (5) is

    R_fs(u) = eps^2 * g * (1 - u)  +  sigma^2 d^2 u / n  +  (c_coupling / n) * eps^2 * g * (1 - u) * u

with g = d / (d - 1).  The correction term is proportional to (1 - u) u, whose
second derivative in u is  -2 * c_coupling * eps^2 * g / n  < 0: the correction is
**concave**, not convex.  Section 3 of the paper says "convex quadratic
correction" and then argues "a convex function on a finite set attains its
minimum at an extreme point", which is false for convex functions and true for
concave ones.  Appendix A and Appendix D of the same paper say "concave" and
give the correct argument.  The endpoint conclusion of Theorem 5 is therefore
right; the main-text wording is a typo.  :func:`correction_curvature` and
:func:`minimiser_is_endpoint` verify the correct statement numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lattice import divisors

__all__ = [
    "param_count",
    "gamma_generic",
    "delta2_generic",
    "delta2_graded",
    "delta2_graded_exact",
    "risk_leading_order",
    "risk_finite_sample",
    "correction_curvature",
    "n_star",
    "eps_star",
    "best_k",
    "minimiser_is_endpoint",
    "graded_window",
    "PAPER_COUPLING",
    "RiskModel",
]

#: Coefficient multiplying ``delta_k^2 / n`` in the coupling term of Proposition 3.
#: The paper asserts ``tr(Pi_k) = p_k = d^2 / k``.  ``symbudget.coupling`` measures it
#: directly; set ``coupling="measured"`` in :class:`RiskModel` to use the measured value.
PAPER_COUPLING = "p_k"


def param_count(d: int, k: int) -> int:
    """``p_k = d^2 / k`` (Proposition 1)."""
    if d % k:
        raise ValueError(f"{k} does not divide {d}")
    return d * d // k


# --------------------------------------------------------------------- defects


def gamma_generic(d: int, k: int | np.ndarray) -> np.ndarray | float:
    """
    Proposition 4:  ``gamma_k = 1 - (d/k - 1)/(d - 1) = d (1 - 1/k) / (d - 1)``.

    ``E_R delta_k^2 = eps^2 gamma_k`` for ``R`` uniform on the unit sphere of ``V_d^perp``.
    """
    k = np.asarray(k, dtype=float)
    return d * (1.0 - 1.0 / k) / (d - 1.0)


def delta2_generic(d: int, k, eps: float) -> np.ndarray | float:
    """Expected squared defect under generic symmetry breaking."""
    return eps**2 * gamma_generic(d, k)


def delta2_graded(d: int, k, k0: int, b: float, c: float) -> np.ndarray | float:
    """
    Proposition 7.  Target ``W* = u_d + b u_{k0} + c u_1`` with
    ``u_d in V_d``, ``u_{k0} in V_{k0} ∩ V_d^perp``, ``u_1 in V_{k0}^perp``:

        delta_k^2 = b^2 + c^2   if  k does not divide-into k0 (i.e. C_k not <= C_{k0})
                  = c^2         if  C_k <= C_{k0} and k < d
                  = 0           if  k = 1

    The paper writes the three cases as ``k < k0`` / ``k0 <= k < d`` / ``k = d``,
    which is correct only when the printed order of divisors happens to agree with
    the lattice order.  The lattice-correct condition is divisibility: a component
    living in ``V_{k0}`` is invisible to ``Pi_k`` unless ``C_k <= C_{k0}``, i.e.
    ``k | k0``.  Appendix C of the paper makes the same point about the measured
    profile being "non-monotone across the printed order".
    """
    scalar = np.ndim(k) == 0
    ks = np.atleast_1d(np.asarray(k, dtype=int))
    out = np.empty(ks.shape, dtype=float)
    for idx, kk in np.ndenumerate(ks):
        if kk == 1:
            out[idx] = 0.0            # V_1 = R^{d x d} represents everything
        elif k0 % kk == 0:            # C_kk <= C_k0 : the k0-component is still representable
            out[idx] = c**2
        else:
            out[idx] = b**2 + c**2
    return float(out[0]) if scalar else out


def _lcm(a: int, b: int) -> int:
    return abs(a * b) // np.gcd(a, b)


def delta2_graded_exact(d: int, k, k0: int, b: float, c: float) -> np.ndarray | float:
    """
    Exact expected graded defect over the **whole** lattice — a refinement of
    Proposition 7.

    Proposition 7 gives a three-case profile (``b^2 + c^2`` / ``c^2`` / ``0``),
    which is exact only at ``k in {1, k0, d}`` and for ``k`` that do not divide
    ``k0``.  For an intermediate ``k | k0`` with ``k < k0`` the architecture
    recovers the ``u_{k0}`` component entirely but only *part* of ``u_1``, so the
    true value is strictly below ``c^2``.  Averaging over unit-norm ``u_{k0}``
    uniform in ``V_{k0} ∩ V_d^perp`` and ``u_1`` uniform in ``V_{k0}^perp``,
    dimension counting gives

        delta_k^2 = b^2 * A_k + c^2 * B_k,
        A_k = (p_{k0} - p_{lcm(k,k0)}) / (p_{k0} - p_d),
        B_k = (p_1 - p_k - p_{k0} + p_{lcm(k,k0)}) / (p_1 - p_{k0}),

    using ``V_k ∩ V_{k0} = V_{lcm(k,k0)}``.  At ``k = 2``, ``k0 = 4``, ``d = 12``,
    ``(b, c) = (0.35, 0.08)`` this gives 0.00427 against Proposition 7's 0.0064 —
    and the paper's own measured value in Appendix C is 0.0043, i.e. the exact
    profile, not the piecewise one.  The three-regime ladder is unaffected.
    """
    scalar = np.ndim(k) == 0
    ks = np.atleast_1d(np.asarray(k, dtype=int))
    p = lambda m: d * d / m
    out = np.empty(ks.shape, dtype=float)
    for idx, kk in np.ndenumerate(ks):
        L = _lcm(int(kk), int(k0))
        A = (p(k0) - p(L)) / (p(k0) - p(d)) if p(k0) > p(d) else 0.0
        B = (p(1) - p(kk) - p(k0) + p(L)) / (p(1) - p(k0)) if p(1) > p(k0) else 0.0
        out[idx] = b**2 * A + c**2 * B
    return float(out[0]) if scalar else out


# ------------------------------------------------------------------------ risk


def risk_leading_order(delta2, d: int, k, sigma: float, n) -> np.ndarray:
    """``R(k) = delta_k^2 + sigma^2 p_k / n``  (Proposition 3, leading order)."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    return np.asarray(delta2, dtype=float) + sigma**2 * (d * d / k) / n


def risk_finite_sample(delta2, d: int, k, sigma: float, n, coupling_coef=None) -> np.ndarray:
    """
    ``R_fs(k) = delta_k^2 + sigma^2 p_k / n + c_k delta_k^2 / n``  (Proposition 3, eq. 3).

    ``coupling_coef`` is ``c_k``.  ``None`` uses the paper's ``c_k = tr(Pi_k) = p_k``.
    Pass a scalar or an array (broadcast against ``k``) to use a measured value.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    p = d * d / k
    c = p if coupling_coef is None else np.asarray(coupling_coef, dtype=float)
    return np.asarray(delta2, dtype=float) * (1.0 + c / n) + sigma**2 * p / n


def correction_curvature(d: int, eps: float, n: float) -> float:
    """
    ``d^2 R_fs / du^2`` for the coupling correction under a generic defect, with
    the paper's ``c_k = d^2 u``.  Negative value == concave correction.

    Substituting ``delta^2 = eps^2 g (1-u)`` and ``c_k = d^2 u`` into the coupling
    term gives ``(eps^2 d^2 g / n) (1-u) u``; its second derivative is
    ``-2 eps^2 d^2 g / n``.
    """
    g = d / (d - 1.0)
    return -2.0 * eps**2 * d**2 * g / n


# -------------------------------------------------------------- phase boundary


def n_star(d: int, sigma: float, eps) -> np.ndarray | float:
    """Corollary 6: ``n* = sigma^2 d (d - 1) / eps^2``."""
    eps = np.asarray(eps, dtype=float)
    return sigma**2 * d * (d - 1.0) / eps**2


def eps_star(d: int, sigma: float, n) -> np.ndarray | float:
    """Inverse of Corollary 6: the defect at which the crossover sits at sample size ``n``."""
    n = np.asarray(n, dtype=float)
    return np.sqrt(sigma**2 * d * (d - 1.0) / n)


def best_k(risks: dict[int, float]) -> int:
    """Argmin of a ``{k: risk}`` map, ties broken towards the more constrained model."""
    finite = {k: r for k, r in risks.items() if np.isfinite(r)}
    if not finite:
        return -1
    best = min(finite.values())
    return max(k for k, r in finite.items() if r <= best + 1e-15)


def minimiser_is_endpoint(d: int, sigma: float, eps: float, n: float, coupling_coef=None) -> bool:
    """Theorem 5: is the minimiser over the whole divisor lattice an endpoint ``k in {1, d}``?"""
    ks = divisors(d)
    risks = {
        k: float(risk_finite_sample(delta2_generic(d, k, eps), d, k, sigma, n,
                                    None if coupling_coef is None else coupling_coef))
        for k in ks
    }
    return best_k(risks) in (1, d)


def graded_window(d: int, k0: int, b: float, c: float, sigma: float) -> tuple[float, float]:
    """
    Proposition 7, eq. (8): the interval of ``n`` on which the intermediate
    architecture ``k0`` is strictly optimal,

        sigma^2 (p_{k0} - p_d) / b^2   <~   n   <~   sigma^2 (p_1 - p_{k0}) / c^2.

    (The paper writes the left denominator as ``b^2 + c^2 - c^2``, i.e. ``b^2``.)
    """
    p1, pk0, pd = param_count(d, 1), param_count(d, k0), param_count(d, d)
    lo = sigma**2 * (pk0 - pd) / (b**2)
    hi = sigma**2 * (p1 - pk0) / (c**2)
    return float(lo), float(hi)


# ------------------------------------------------------------------ container


@dataclass
class RiskModel:
    """
    Convenience wrapper bundling ``(d, sigma)`` with a defect profile.

    ``delta2`` maps ``k -> delta_k^2``; use :func:`delta2_generic` or
    :func:`delta2_graded`, or a measured profile.
    """

    d: int
    sigma: float
    delta2: dict[int, float]
    coupling: dict[int, float] | None = None

    @classmethod
    def generic(cls, d: int, sigma: float, eps: float, coupling: dict[int, float] | None = None) -> "RiskModel":
        return cls(d, sigma, {k: float(delta2_generic(d, k, eps)) for k in divisors(d)}, coupling)

    @classmethod
    def graded(cls, d: int, sigma: float, k0: int, b: float, c: float,
               coupling: dict[int, float] | None = None) -> "RiskModel":
        return cls(d, sigma, {k: float(delta2_graded(d, k, k0, b, c)) for k in divisors(d)}, coupling)

    def risks(self, n: float, finite_sample: bool = True) -> dict[int, float]:
        out = {}
        for k, d2 in self.delta2.items():
            if finite_sample:
                cc = None if self.coupling is None else self.coupling.get(k)
                out[k] = float(risk_finite_sample(d2, self.d, k, self.sigma, n, cc))
            else:
                out[k] = float(risk_leading_order(d2, self.d, k, self.sigma, n))
        return out

    def argmin(self, n: float, finite_sample: bool = True) -> int:
        return best_k(self.risks(n, finite_sample))

    def crossover(self, lo: float = 1.0, hi: float = 1e7, finite_sample: bool = True) -> float:
        """Smallest ``n`` (by bisection on the sign of R(k=1) - R(k=d)) at which dense overtakes equivariant."""
        f = lambda n: self.risks(n, finite_sample)[1] - self.risks(n, finite_sample)[self.d]
        if f(hi) > 0:
            return float("inf")
        if f(lo) < 0:
            return float(lo)
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if f(mid) > 0:
                lo = mid
            else:
                hi = mid
        return float(np.sqrt(lo * hi))
