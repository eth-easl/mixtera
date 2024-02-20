"""
This submodule contains code for calculating new properties of datasets
"""

from .executor import PropertyCalculationExecutor  # noqa: F401
from .local_executor import LocalPropertyCalculationExecutor  # noqa: F401

__all__ = ["PropertyCalculationExecutor", "LocalPropertyCalculationExecutor"]
