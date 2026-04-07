"""Quadrature method implementations."""

from flashquad.methods.trapezoid import trapz
from flashquad.methods.simpson import simpson
from flashquad.methods.boole import booles
from flashquad.methods.gauss import gauss
from flashquad.methods.mc import mc
from flashquad.methods.adpmc import adpmc

__all__ = ["trapz", "simpson", "booles", "gauss", "mc", "adpmc"]
