"""
Significant Factor Analyzer Module

Identifies which features (sensors/variables) contribute most
to detected anomalies for root cause analysis.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


@dataclass
class FactorContribution:
    """Container for factor contribution analysis."""
    feature_names: List[str]
    contribution_scores: np.ndarray
    contribution_percentages: np.ndarray
    ranked_features: List[Tuple[str, float]]
    top_contributors: List[str]
    deviation_direction: np.ndarray  # Positive or negative deviation


@dataclass
class AnomalyDiagnostics:
    """Detailed diagnostics for anomaly points."""
    point_index: int
    anomaly_score: float
    category: str
    contributing_features: List[Tuple[str, float, str]]  # (name, contribution, direction)
    deviation_summary: str
    recommended_actions: List[str]


class SignificantFactorAnalyzer:
    """
    Analyzes which features contribute most to anomalies.

    Uses multiple methods to determine feature importance:
    1. Residual decomposition
    2. Feature importance (Random Forest)
    3. Sensitivity analysis
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize the analyzer.

        Args:
            feature_names: Names of features/sensors. If None, generic names are used.
        """
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        self.baseline_stats: Optional[Dict] = None

    def set_feature_names(self, names: List[str]) -> None:
        """Set feature names for reporting."""
        self.feature_names = names

    def fit_baseline(self, normal_data: np.ndarray) -> 'SignificantFactorAnalyzer':
        """
        Fit baseline statistics from normal operating data.

        Args:
            normal_data: Data representing normal operating conditions

        Returns:
            Self
        """
        n_features = normal_data.shape[1]

        if self.feature_names is None:
            self.feature_names = [f"Feature_{i+1}" for i in range(n_features)]

        self.baseline_stats = {
            'mean': np.mean(normal_data, axis=0),
            'std': np.std(normal_data, axis=0),
            'min': np.min(normal_data, axis=0),
            'max': np.max(normal_data, axis=0),
            'median': np.median(normal_data, axis=0),
            'q25': np.percentile(normal_data, 25, axis=0),
            'q75': np.percentile(normal_data, 75, axis=0),
            'correlation': np.corrcoef(normal_data.T) if n_features > 1 else np.array([[1.0]])
        }

        self.scaler.fit(normal_data)

        return self

    def analyze_contributions(self, data: np.ndarray,
                               feature_residuals: np.ndarray,
                               point_residuals: np.ndarray) -> FactorContribution:
        """
        Analyze which features contribute most to the overall residuals.

        Args:
            data: Current data points
            feature_residuals: Per-feature residuals from ResidualCalculator
            point_residuals: Overall point residuals

        Returns:
            FactorContribution with analysis results
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        n_features = feature_residuals.shape[1]

        # Method 1: Direct residual contribution
        # Absolute contribution of each feature to total residual
        abs_feature_residuals = np.abs(feature_residuals)
        mean_abs_contributions = np.mean(abs_feature_residuals, axis=0)

        # Normalize by baseline std for fair comparison
        baseline_std = self.baseline_stats['std']
        baseline_std[baseline_std == 0] = 1  # Avoid division by zero
        normalized_contributions = mean_abs_contributions / baseline_std

        # Calculate percentages
        total_contribution = normalized_contributions.sum()
        if total_contribution > 0:
            contribution_percentages = (normalized_contributions / total_contribution) * 100
        else:
            contribution_percentages = np.zeros(n_features)

        # Determine deviation direction (positive = above normal, negative = below)
        mean_residuals = np.mean(feature_residuals, axis=0)
        deviation_direction = np.sign(mean_residuals)

        # Rank features by contribution
        ranked_indices = np.argsort(normalized_contributions)[::-1]
        ranked_features = [
            (self.feature_names[i], contribution_percentages[i])
            for i in ranked_indices
        ]

        # Get top contributors (those contributing > 10% or top 3)
        top_threshold = 10.0
        top_contributors = [
            name for name, pct in ranked_features
            if pct > top_threshold
        ]
        if len(top_contributors) == 0:
            top_contributors = [name for name, _ in ranked_features[:3]]

        return FactorContribution(
            feature_names=self.feature_names,
            contribution_scores=normalized_contributions,
            contribution_percentages=contribution_percentages,
            ranked_features=ranked_features,
            top_contributors=top_contributors,
            deviation_direction=deviation_direction
        )

    def analyze_point(self, point_idx: int,
                      data: np.ndarray,
                      feature_residuals: np.ndarray,
                      point_residuals: np.ndarray,
                      anomaly_categories: np.ndarray) -> AnomalyDiagnostics:
        """
        Analyze a specific anomaly point in detail.

        Args:
            point_idx: Index of the point to analyze
            data: All data points
            feature_residuals: Per-feature residuals
            point_residuals: Overall point residuals
            anomaly_categories: Anomaly categories for all points

        Returns:
            AnomalyDiagnostics with detailed analysis
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        point_data = data[point_idx]
        point_feat_residuals = feature_residuals[point_idx]
        anomaly_score = point_residuals[point_idx]
        category_num = anomaly_categories[point_idx]

        category_names = {0: 'Normal', 1: 'Low', 2: 'Medium', 3: 'High'}
        category = category_names.get(category_num, 'Unknown')

        # Calculate contribution for this specific point
        baseline_std = self.baseline_stats['std'].copy()
        baseline_std[baseline_std == 0] = 1

        abs_residuals = np.abs(point_feat_residuals)
        normalized = abs_residuals / baseline_std
        total = normalized.sum()

        if total > 0:
            contributions = (normalized / total) * 100
        else:
            contributions = np.zeros(len(normalized))

        # Sort by contribution
        sorted_indices = np.argsort(contributions)[::-1]

        contributing_features = []
        for idx in sorted_indices:
            if contributions[idx] > 5:  # Only significant contributions
                direction = "above" if point_feat_residuals[idx] > 0 else "below"
                contributing_features.append((
                    self.feature_names[idx],
                    contributions[idx],
                    direction
                ))

        # Generate deviation summary
        deviation_summary = self._generate_deviation_summary(
            point_data, contributing_features
        )

        # Generate recommended actions
        recommended_actions = self._generate_recommendations(
            category, contributing_features
        )

        return AnomalyDiagnostics(
            point_index=point_idx,
            anomaly_score=anomaly_score,
            category=category,
            contributing_features=contributing_features,
            deviation_summary=deviation_summary,
            recommended_actions=recommended_actions
        )

    def _generate_deviation_summary(self, point_data: np.ndarray,
                                     contributing_features: List[Tuple]) -> str:
        """Generate a human-readable deviation summary."""
        if not contributing_features:
            return "No significant deviations detected."

        parts = []
        for name, contribution, direction in contributing_features[:3]:
            feature_idx = self.feature_names.index(name)
            current_val = point_data[feature_idx]
            baseline_mean = self.baseline_stats['mean'][feature_idx]
            baseline_std = self.baseline_stats['std'][feature_idx]

            if baseline_std > 0:
                z_score = (current_val - baseline_mean) / baseline_std
                parts.append(
                    f"{name}: {current_val:.2f} ({z_score:+.1f} sigma {direction} normal)"
                )
            else:
                parts.append(f"{name}: {current_val:.2f} ({direction} normal)")

        return "; ".join(parts)

    def _generate_recommendations(self, category: str,
                                   contributing_features: List[Tuple]) -> List[str]:
        """Generate recommended actions based on anomaly analysis."""
        recommendations = []

        if category == 'Normal':
            return ["No action required - operating within normal parameters."]

        if category == 'Low':
            recommendations.append("Monitor the following sensors more frequently.")
        elif category == 'Medium':
            recommendations.append("Schedule inspection of related components.")
            recommendations.append("Review historical trends for these parameters.")
        elif category == 'High':
            recommendations.append("URGENT: Immediate inspection recommended.")
            recommendations.append("Consider reducing operational load.")
            recommendations.append("Prepare for potential maintenance intervention.")

        # Add specific recommendations based on contributing features
        for name, contribution, direction in contributing_features[:3]:
            if contribution > 20:
                if direction == "above":
                    recommendations.append(
                        f"Check {name}: value is significantly above normal range."
                    )
                else:
                    recommendations.append(
                        f"Check {name}: value is significantly below normal range."
                    )

        return recommendations

    def get_feature_importance_ml(self, data: np.ndarray,
                                   residuals: np.ndarray) -> Dict[str, float]:
        """
        Use machine learning to determine feature importance.

        Trains a Random Forest to predict residuals from features
        and extracts feature importances.
        """
        if len(data) < 20:
            # Not enough data for ML approach
            return {name: 1.0 / len(self.feature_names)
                    for name in self.feature_names}

        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(data, residuals)

        importances = rf.feature_importances_

        return {
            self.feature_names[i]: importances[i]
            for i in range(len(self.feature_names))
        }

    def correlation_breakdown(self, data: np.ndarray) -> Dict[str, Dict]:
        """
        Analyze how correlations between features have changed.

        Useful for detecting sensor drift or relationship changes.
        """
        if self.baseline_stats is None:
            raise ValueError("Baseline not fitted. Call fit_baseline() first.")

        n_features = data.shape[1]
        current_corr = np.corrcoef(data.T) if n_features > 1 else np.array([[1.0]])
        baseline_corr = self.baseline_stats['correlation']

        # Find significant correlation changes
        corr_diff = current_corr - baseline_corr
        significant_changes = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                diff = corr_diff[i, j]
                if abs(diff) > 0.2:  # Significant change threshold
                    significant_changes.append({
                        'feature_1': self.feature_names[i],
                        'feature_2': self.feature_names[j],
                        'baseline_correlation': baseline_corr[i, j],
                        'current_correlation': current_corr[i, j],
                        'change': diff
                    })

        return {
            'baseline_correlation': baseline_corr,
            'current_correlation': current_corr,
            'correlation_diff': corr_diff,
            'significant_changes': significant_changes
        }

    def get_trend_analysis(self, time_series_data: np.ndarray,
                           window_size: int = 10) -> Dict[str, np.ndarray]:
        """
        Analyze trends in the time series data.

        Args:
            time_series_data: Time series with shape (n_timesteps, n_features)
            window_size: Window size for moving average

        Returns:
            Dictionary with trend information for each feature
        """
        n_timesteps, n_features = time_series_data.shape

        trends = {}
        for i in range(n_features):
            feature_data = time_series_data[:, i]
            name = self.feature_names[i]

            # Calculate moving average
            if n_timesteps >= window_size:
                moving_avg = np.convolve(
                    feature_data,
                    np.ones(window_size) / window_size,
                    mode='valid'
                )
            else:
                moving_avg = feature_data

            # Calculate trend direction
            if len(moving_avg) > 1:
                trend_slope = np.polyfit(range(len(moving_avg)), moving_avg, 1)[0]
            else:
                trend_slope = 0

            trends[name] = {
                'moving_average': moving_avg,
                'trend_slope': trend_slope,
                'trend_direction': 'increasing' if trend_slope > 0.01 else (
                    'decreasing' if trend_slope < -0.01 else 'stable'
                ),
                'current_value': feature_data[-1],
                'mean_value': np.mean(feature_data),
                'volatility': np.std(feature_data)
            }

        return trends
