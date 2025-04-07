"""
DataFlow: A lightweight Python library for building composable data processing pipelines.
"""

__version__ = "0.1.0"

# Import core components for easy access
from dataflow.core import Pipeline, Step, step
import dataflow.transforms as transforms
import dataflow.validators as validators
import dataflow.utils as utils

# Make key components available at the top level
__all__ = [
    "Pipeline",
    "Step",
    "step",
    "transforms",
    "validators",
    "utils",
]