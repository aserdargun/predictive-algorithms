"""
Predictive Maintenance Tool using Clustering Algorithms

This package provides a clustering-based predictive maintenance solution
using K-means, DBSCAN, OPTICS, SOM, and LSH algorithms to detect anomalies
and calculate Overall Model Residuals (OMR).
"""

from .predictive_maintenance import PredictiveMaintenanceTool
from .clustering import ClusteringEngine
from .residual_calculator import ResidualCalculator
from .factor_analyzer import SignificantFactorAnalyzer

__version__ = "1.0.0"
__all__ = [
    "PredictiveMaintenanceTool",
    "ClusteringEngine",
    "ResidualCalculator",
    "SignificantFactorAnalyzer"
]
