"""
symbudget -- a complete implementation of "The Symmetry Budget: How Much Group
Structure Should a Network Architecture Assume?"

Layout
------
``lattice``      subgroup lattice of C_d, orbit bases, projections (Prop. 1-2)
``theory``       closed-form defects, risk, phase boundary (Prop. 3-4, 7, Thm. 5, Cor. 6)
``data``         the data model of eq. (2), generic and graded targets, sufficient statistics
``estimators``   constrained least squares, excess risk, checkpoint store
``nonlinear``    two-layer tanh control with projected gradient descent
``nonabelian``   Proposition 8 audit over finite groups and their representations
``config``       experiment presets (paper / quick / smoke)
``experiments``  the nine experiment stages
``figures``      figure generation
``validate``     claim-by-claim validation harness

Quick start
-----------
>>> from symbudget import lattice, theory
>>> lat = lattice.lattice(12)
>>> [lat.p[k] for k in lat.ks]
[144, 72, 48, 36, 24, 12]
>>> float(theory.n_star(12, 0.5, 0.2))
825.0
"""

from . import config, data, estimators, figures, lattice, nonabelian, nonlinear, theory, validate

__version__ = "1.0.0"
__all__ = ["config", "data", "estimators", "figures", "lattice", "nonabelian",
           "nonlinear", "theory", "validate", "__version__"]
