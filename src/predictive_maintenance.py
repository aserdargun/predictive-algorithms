"""
Predictive Maintenance Tool

Main class that integrates clustering, residual calculation,
and factor analysis for predictive maintenance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
import warnings

from .clustering import ClusteringEngine, ClusterResult
from .residual_calculator import ResidualCalculator, MultiScaleResidualCalculator, ResidualResult
from .factor_analyzer import SignificantFactorAnalyzer, FactorContribution, AnomalyDiagnostics


@dataclass
class MaintenanceReport:
    """Complete predictive maintenance analysis report."""
    timestamp: str
    n_samples: int
    n_features: int
    feature_names: List[str]

    # Overall metrics
    overall_model_residual: float
    omr_category: str
    anomaly_rate: float

    # Clustering results
    clustering_summary: Dict[str, Dict]
    best_clustering_algorithm: str

    # Residual analysis
    residual_summary: Dict
    anomaly_indices: np.ndarray
    anomaly_categories: np.ndarray

    # Factor analysis
    factor_contributions: FactorContribution
    top_contributing_factors: List[Tuple[str, float]]

    # Detailed diagnostics for anomalies
    anomaly_diagnostics: List[AnomalyDiagnostics] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'overall_model_residual': float(self.overall_model_residual),
            'omr_category': self.omr_category,
            'anomaly_rate': float(self.anomaly_rate),
            'clustering_summary': self.clustering_summary,
            'best_clustering_algorithm': self.best_clustering_algorithm,
            'residual_summary': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in self.residual_summary.items()
            },
            'top_contributing_factors': [
                (name, float(score)) for name, score in self.top_contributing_factors
            ]
        }

    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class PredictiveMaintenanceTool:
    """
    Clustering-based Predictive Maintenance Tool.

    Uses multiple clustering algorithms (K-means, DBSCAN, OPTICS, SOM, LSH)
    combined with nearest-neighbor residual analysis to detect anomalies
    and identify significant contributing factors.

    Inspired by AVEVA Predictive Analytics approach.
    """

    MAX_TIMESERIES = 15  # Maximum number of time series (features) supported

    def __init__(self,
                 feature_names: Optional[List[str]] = None,
                 n_neighbors: int = 5,
                 multi_scale: bool = True,
                 algorithms: Optional[List[str]] = None,
                 random_state: int = 42):
        """
        Initialize the Predictive Maintenance Tool.

        Args:
            feature_names: Names of features/sensors (max 15)
            n_neighbors: Number of neighbors for residual calculation
            multi_scale: Use multi-scale residual calculation
            algorithms: Clustering algorithms to use
            random_state: Random seed for reproducibility
        """
        self.feature_names = feature_names
        self.n_neighbors = n_neighbors
        self.multi_scale = multi_scale
        self.algorithms = algorithms or ['kmeans', 'dbscan', 'optics', 'som', 'lsh']
        self.random_state = random_state

        self.clustering_engine = ClusteringEngine(random_state=random_state)

        if multi_scale:
            self.residual_calculator = MultiScaleResidualCalculator(
                k_values=[3, 5, 10, min(20, n_neighbors * 2)]
            )
        else:
            self.residual_calculator = ResidualCalculator(n_neighbors=n_neighbors)

        self.factor_analyzer = SignificantFactorAnalyzer(feature_names=feature_names)

        self.is_fitted = False
        self.training_data: Optional[np.ndarray] = None

    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        """Validate and prepare input data."""
        if isinstance(data, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = list(data.columns)
            data = data.values

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        n_features = data.shape[1]
        if n_features > self.MAX_TIMESERIES:
            warnings.warn(
                f"Data has {n_features} features, but max is {self.MAX_TIMESERIES}. "
                f"Only first {self.MAX_TIMESERIES} features will be used."
            )
            data = data[:, :self.MAX_TIMESERIES]

        if self.feature_names is None:
            self.feature_names = [f"Sensor_{i+1}" for i in range(data.shape[1])]
        elif len(self.feature_names) < data.shape[1]:
            # Extend feature names if needed
            for i in range(len(self.feature_names), data.shape[1]):
                self.feature_names.append(f"Sensor_{i+1}")

        # Handle NaN values
        if np.isnan(data).any():
            warnings.warn("Data contains NaN values. They will be interpolated.")
            data = self._interpolate_nans(data)

        return data

    def _interpolate_nans(self, data: np.ndarray) -> np.ndarray:
        """Interpolate NaN values in the data."""
        result = data.copy()
        for col in range(data.shape[1]):
            mask = np.isnan(result[:, col])
            if mask.all():
                result[:, col] = 0
            elif mask.any():
                valid_indices = np.where(~mask)[0]
                result[mask, col] = np.interp(
                    np.where(mask)[0],
                    valid_indices,
                    result[valid_indices, col]
                )
        return result

    def fit(self, training_data: Union[np.ndarray, pd.DataFrame],
            clustering_params: Optional[Dict] = None) -> 'PredictiveMaintenanceTool':
        """
        Fit the model on training data (normal operating conditions).

        Args:
            training_data: Historical data representing normal operations
                          Shape: (n_samples, n_features) where n_features <= 15
            clustering_params: Optional parameters for clustering algorithms

        Returns:
            Self
        """
        data = self._validate_data(training_data)
        self.training_data = data

        # Fit clustering engine
        self.clustering_engine.fit(
            data,
            algorithms=self.algorithms,
            params=clustering_params
        )

        # Fit residual calculator
        self.residual_calculator.fit(data)

        # Fit factor analyzer baseline
        self.factor_analyzer.set_feature_names(self.feature_names[:data.shape[1]])
        self.factor_analyzer.fit_baseline(data)

        self.is_fitted = True
        return self

    def analyze(self, data: Union[np.ndarray, pd.DataFrame],
                detailed_diagnostics: bool = True) -> MaintenanceReport:
        """
        Analyze new data for anomalies.

        Args:
            data: New data to analyze. Shape: (n_samples, n_features)
            detailed_diagnostics: Generate detailed diagnostics for anomaly points

        Returns:
            MaintenanceReport with complete analysis
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        data = self._validate_data(data)

        # Get cluster labels from ensemble
        cluster_labels = self.clustering_engine.get_ensemble_labels()

        # Calculate residuals
        residual_result = self.residual_calculator.calculate_residuals(
            data, cluster_labels
        )

        # Analyze contributing factors
        factor_contributions = self.factor_analyzer.analyze_contributions(
            data,
            residual_result.feature_residuals,
            residual_result.point_residuals
        )

        # Generate clustering summary
        clustering_summary = self._generate_clustering_summary()

        # Determine best algorithm
        best_algorithm = self._select_best_algorithm()

        # Calculate anomaly rate
        anomaly_indices = np.where(residual_result.anomaly_categories > 0)[0]
        anomaly_rate = len(anomaly_indices) / len(data)

        # Categorize OMR
        omr_category = self._categorize_omr(residual_result.overall_model_residual)

        # Generate residual summary
        residual_summary = {
            'mean_residual': np.mean(residual_result.point_residuals),
            'std_residual': np.std(residual_result.point_residuals),
            'max_residual': np.max(residual_result.point_residuals),
            'min_residual': np.min(residual_result.point_residuals),
            'threshold_low': residual_result.threshold_low,
            'threshold_medium': residual_result.threshold_medium,
            'threshold_high': residual_result.threshold_high,
            'n_low_anomalies': np.sum(residual_result.anomaly_categories == 1),
            'n_medium_anomalies': np.sum(residual_result.anomaly_categories == 2),
            'n_high_anomalies': np.sum(residual_result.anomaly_categories == 3)
        }

        # Get top contributing factors
        top_factors = factor_contributions.ranked_features[:5]

        # Generate detailed diagnostics for anomaly points
        anomaly_diagnostics = []
        if detailed_diagnostics:
            # Analyze top anomalies (by residual)
            high_anomaly_indices = np.where(residual_result.anomaly_categories >= 2)[0]
            for idx in high_anomaly_indices[:10]:  # Limit to top 10
                diag = self.factor_analyzer.analyze_point(
                    idx, data,
                    residual_result.feature_residuals,
                    residual_result.point_residuals,
                    residual_result.anomaly_categories
                )
                anomaly_diagnostics.append(diag)

        from datetime import datetime

        return MaintenanceReport(
            timestamp=datetime.now().isoformat(),
            n_samples=len(data),
            n_features=data.shape[1],
            feature_names=self.feature_names[:data.shape[1]],
            overall_model_residual=residual_result.overall_model_residual,
            omr_category=omr_category,
            anomaly_rate=anomaly_rate,
            clustering_summary=clustering_summary,
            best_clustering_algorithm=best_algorithm,
            residual_summary=residual_summary,
            anomaly_indices=anomaly_indices,
            anomaly_categories=residual_result.anomaly_categories,
            factor_contributions=factor_contributions,
            top_contributing_factors=top_factors,
            anomaly_diagnostics=anomaly_diagnostics
        )

    def _generate_clustering_summary(self) -> Dict[str, Dict]:
        """Generate summary of clustering results."""
        summary = {}
        for alg, result in self.clustering_engine.get_all_results().items():
            summary[alg] = {
                'n_clusters': result.n_clusters,
                'n_noise_points': len(result.noise_points),
                'silhouette_score': result.silhouette_score,
                'parameters': result.parameters
            }
        return summary

    def _select_best_algorithm(self) -> str:
        """Select the best clustering algorithm based on silhouette score."""
        best_alg = None
        best_score = -1

        for alg, result in self.clustering_engine.get_all_results().items():
            if result.silhouette_score is not None:
                if result.silhouette_score > best_score:
                    best_score = result.silhouette_score
                    best_alg = alg

        return best_alg or 'kmeans'

    def _categorize_omr(self, omr: float) -> str:
        """Categorize the Overall Model Residual."""
        # These thresholds should be calibrated based on the specific application
        if omr < 0.5:
            return 'Normal'
        elif omr < 1.0:
            return 'Low'
        elif omr < 2.0:
            return 'Medium'
        else:
            return 'High'

    def get_anomaly_scores(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get anomaly scores for data points.

        Returns scores on a 0-100 scale where higher = more anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        data = self._validate_data(data)
        cluster_labels = self.clustering_engine.get_ensemble_labels()
        result = self.residual_calculator.calculate_residuals(data, cluster_labels)

        return result.anomaly_scores

    def get_residuals(self, data: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, float]:
        """
        Get point residuals and overall model residual.

        Returns:
            Tuple of (point_residuals, overall_model_residual)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        data = self._validate_data(data)
        cluster_labels = self.clustering_engine.get_ensemble_labels()
        result = self.residual_calculator.calculate_residuals(data, cluster_labels)

        return result.point_residuals, result.overall_model_residual

    def get_cluster_labels(self, algorithm: Optional[str] = None) -> np.ndarray:
        """
        Get cluster labels from a specific algorithm or ensemble.

        Args:
            algorithm: Specific algorithm name, or None for ensemble
        """
        if algorithm is None:
            return self.clustering_engine.get_ensemble_labels()
        else:
            return self.clustering_engine.get_result(algorithm).labels

    def print_report(self, report: MaintenanceReport) -> str:
        """Generate a formatted string report."""
        lines = [
            "=" * 60,
            "PREDICTIVE MAINTENANCE ANALYSIS REPORT",
            "=" * 60,
            f"Timestamp: {report.timestamp}",
            f"Samples analyzed: {report.n_samples}",
            f"Features: {report.n_features}",
            "",
            "-" * 40,
            "OVERALL STATUS",
            "-" * 40,
            f"Overall Model Residual (OMR): {report.overall_model_residual:.4f}",
            f"OMR Category: {report.omr_category}",
            f"Anomaly Rate: {report.anomaly_rate * 100:.1f}%",
            "",
            "-" * 40,
            "CLUSTERING ANALYSIS",
            "-" * 40,
            f"Best Algorithm: {report.best_clustering_algorithm}",
        ]

        for alg, info in report.clustering_summary.items():
            sil = info['silhouette_score']
            sil_str = f"{sil:.3f}" if sil is not None else "N/A"
            lines.append(
                f"  {alg}: {info['n_clusters']} clusters, "
                f"silhouette={sil_str}, noise={info['n_noise_points']}"
            )

        lines.extend([
            "",
            "-" * 40,
            "RESIDUAL ANALYSIS",
            "-" * 40,
            f"Mean Residual: {report.residual_summary['mean_residual']:.4f}",
            f"Std Residual: {report.residual_summary['std_residual']:.4f}",
            f"Low Anomalies: {report.residual_summary['n_low_anomalies']}",
            f"Medium Anomalies: {report.residual_summary['n_medium_anomalies']}",
            f"High Anomalies: {report.residual_summary['n_high_anomalies']}",
            "",
            "-" * 40,
            "TOP CONTRIBUTING FACTORS",
            "-" * 40,
        ])

        for name, score in report.top_contributing_factors:
            lines.append(f"  {name}: {score:.1f}%")

        if report.anomaly_diagnostics:
            lines.extend([
                "",
                "-" * 40,
                "ANOMALY DIAGNOSTICS",
                "-" * 40,
            ])
            for diag in report.anomaly_diagnostics[:5]:
                lines.append(f"\nPoint {diag.point_index} ({diag.category} anomaly):")
                lines.append(f"  Score: {diag.anomaly_score:.4f}")
                lines.append(f"  Summary: {diag.deviation_summary}")
                lines.append("  Recommendations:")
                for rec in diag.recommended_actions[:2]:
                    lines.append(f"    - {rec}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def load_timeseries_data(file_path: str,
                         timestamp_col: Optional[str] = None,
                         feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    Load time series data from a CSV file.

    Args:
        file_path: Path to CSV file
        timestamp_col: Name of timestamp column (will be excluded from features)
        feature_cols: Specific feature columns to use

    Returns:
        Tuple of (data array, feature names)
    """
    df = pd.read_csv(file_path)

    if timestamp_col and timestamp_col in df.columns:
        df = df.drop(columns=[timestamp_col])

    if feature_cols:
        df = df[feature_cols]

    # Limit to 15 features
    if len(df.columns) > 15:
        warnings.warn(f"Data has {len(df.columns)} columns. Using first 15.")
        df = df.iloc[:, :15]

    return df.values, list(df.columns)
