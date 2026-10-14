"""Curve fitting and robust regression used by the pyhctsa operations."""
from .curves import fit_curve, fit_exp, goodness_of_fit, polyfit
from .robust_regression import BisquareIRLS, robust_linear_fit

__all__ = ["fit_curve", "fit_exp", "goodness_of_fit", "polyfit", "BisquareIRLS", "robust_linear_fit"]
