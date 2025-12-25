"""
Residual Calculator Module

Calculates model residuals using nearest neighbor approaches
for predictive maintenance anomaly detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class ResidualResult:
    """Container for residual calculation results."""
    point_residuals: np.ndarray
    overall_model_residual: float
    feature_residuals: np.ndarray
    anomaly_scores: np.ndarray
    threshold_low: float
    threshold_medium: float
    threshold_high: float
    anomaly_categories: np.ndarray


class ResidualCalculator:
    """
    Calculates residuals using K-Nearest Neighbors approach.

    The residual for each point is computed as the difference between
    the actual value and the predicted value based on its neighbors.
    This is similar to the AVEVA Predictive Analytics approach.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = 'distance',
                 algorithm: str = 'auto'):
        """
        Initialize the Residual Calculator.

        Args:
            n_neighbors: Number of neighbors to use for prediction
            weights: Weight function ('uniform' or 'distance')
            algorithm: Algorithm for NearestNeighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.scaler = StandardScaler()
        self.nn_model: Optional[NearestNeighbors] = None
        self.training_data: Optional[np.ndarray] = None
        self.training_data_scaled: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> 'ResidualCalculator':
        """
        Fit the model on training data (normal operating conditions).

        Args:
            data: Training data array of shape (n_samples, n_features)

        Returns:
            Self
        """
        self.training_data = data.copy()
        self.training_data_scaled = self.scaler.fit_transform(data)

        # Ensure n_neighbors doesn't exceed available samples
        k = min(self.n_neighbors, len(data) - 1)

        self.nn_model = NearestNeighbors(
            n_neighbors=k + 1,  # +1 because point itself will be included
            algorithm=self.algorithm
        )
        self.nn_model.fit(self.training_data_scaled)

        return self

    def calculate_residuals(self, data: np.ndarray,
                            cluster_labels: Optional[np.ndarray] = None) -> ResidualResult:
        """
        Calculate residuals for new data points.

        The residual is computed as the difference between the actual value
        and the weighted average of its K nearest neighbors.

        Args:
            data: Data to calculate residuals for
            cluster_labels: Optional cluster labels for cluster-aware residuals

        Returns:
            ResidualResult containing all residual metrics
        """
        if self.nn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        data_scaled = self.scaler.transform(data)
        n_samples, n_features = data.shape

        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(data_scaled)

        # Calculate predictions based on neighbors
        point_residuals = np.zeros(n_samples)
        feature_residuals = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            # Get neighbor indices (excluding self if present in training data)
            neighbor_idx = indices[i][1:] if distances[i][0] < 1e-10 else indices[i][:-1]
            neighbor_distances = distances[i][1:] if distances[i][0] < 1e-10 else distances[i][:-1]

            # Calculate weights
            if self.weights == 'distance':
                # Avoid division by zero
                weights = 1.0 / (neighbor_distances + 1e-10)
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(neighbor_idx)) / len(neighbor_idx)

            # Predicted value is weighted average of neighbors
            neighbor_values = self.training_data[neighbor_idx]
            predicted = np.average(neighbor_values, axis=0, weights=weights)

            # Residual is difference between actual and predicted
            residual_vector = data[i] - predicted
            feature_residuals[i] = residual_vector

            # Point residual is L2 norm of the residual vector
            point_residuals[i] = np.linalg.norm(residual_vector)

        # Apply cluster-aware adjustment if labels provided
        if cluster_labels is not None:
            point_residuals = self._apply_cluster_adjustment(
                point_residuals, cluster_labels, data_scaled
            )

        # Calculate Overall Model Residual (OMR)
        omr = self._calculate_omr(point_residuals)

        # Calculate anomaly scores (normalized residuals)
        anomaly_scores = self._calculate_anomaly_scores(point_residuals)

        # Determine thresholds
        thresholds = self._calculate_thresholds(point_residuals)

        # Categorize anomalies
        categories = self._categorize_anomalies(point_residuals, thresholds)

        return ResidualResult(
            point_residuals=point_residuals,
            overall_model_residual=omr,
            feature_residuals=feature_residuals,
            anomaly_scores=anomaly_scores,
            threshold_low=thresholds['low'],
            threshold_medium=thresholds['medium'],
            threshold_high=thresholds['high'],
            anomaly_categories=categories
        )

    def _apply_cluster_adjustment(self, residuals: np.ndarray,
                                   labels: np.ndarray,
                                   data_scaled: np.ndarray) -> np.ndarray:
        """
        Adjust residuals based on cluster membership.

        Points that are noise (label=-1) or far from cluster center
        get higher residuals.
        """
        adjusted = residuals.copy()

        # Noise points get penalty
        noise_mask = labels == -1
        if noise_mask.any():
            noise_factor = 1.5
            adjusted[noise_mask] *= noise_factor

        # Adjust based on distance to cluster center
        unique_labels = set(labels)
        unique_labels.discard(-1)

        for label in unique_labels:
            mask = labels == label
            cluster_data = data_scaled[mask]
            cluster_center = cluster_data.mean(axis=0)

            distances_to_center = np.linalg.norm(cluster_data - cluster_center, axis=1)

            # Normalize distances within cluster
            if distances_to_center.std() > 0:
                normalized_dist = (distances_to_center - distances_to_center.mean()) / distances_to_center.std()
                adjustment = 1 + 0.1 * np.clip(normalized_dist, 0, 3)
                adjusted[mask] *= adjustment

        return adjusted

    def _calculate_omr(self, point_residuals: np.ndarray) -> float:
        """
        Calculate Overall Model Residual (OMR).

        OMR is a single metric representing the overall model performance.
        It combines mean residual with a penalty for high-residual outliers.
        """
        # Base OMR is the mean residual
        mean_residual = np.mean(point_residuals)

        # Add penalty for outliers (points above 95th percentile)
        p95 = np.percentile(point_residuals, 95)
        outlier_mask = point_residuals > p95
        outlier_penalty = np.mean(point_residuals[outlier_mask]) if outlier_mask.any() else 0

        # Weighted combination
        omr = 0.7 * mean_residual + 0.3 * outlier_penalty

        return omr

    def _calculate_anomaly_scores(self, point_residuals: np.ndarray) -> np.ndarray:
        """
        Calculate normalized anomaly scores (0-100 scale).

        Higher score indicates higher likelihood of anomaly.
        """
        # Use robust scaling based on median and IQR
        median = np.median(point_residuals)
        q25, q75 = np.percentile(point_residuals, [25, 75])
        iqr = q75 - q25

        if iqr > 0:
            normalized = (point_residuals - median) / iqr
        else:
            normalized = point_residuals - median

        # Convert to 0-100 scale using sigmoid-like transformation
        scores = 100 / (1 + np.exp(-normalized))

        return scores

    def _calculate_thresholds(self, point_residuals: np.ndarray) -> Dict[str, float]:
        """
        Calculate threshold values for anomaly categorization.

        Uses statistical methods similar to control chart limits.
        """
        mean_res = np.mean(point_residuals)
        std_res = np.std(point_residuals)

        # Also consider percentile-based thresholds
        p75 = np.percentile(point_residuals, 75)
        p90 = np.percentile(point_residuals, 90)
        p99 = np.percentile(point_residuals, 99)

        return {
            'low': max(mean_res + 1.5 * std_res, p75),
            'medium': max(mean_res + 2.5 * std_res, p90),
            'high': max(mean_res + 3.5 * std_res, p99)
        }

    def _categorize_anomalies(self, point_residuals: np.ndarray,
                               thresholds: Dict[str, float]) -> np.ndarray:
        """
        Categorize each point as normal, low, medium, or high anomaly.

        Categories:
        - 0: Normal
        - 1: Low anomaly
        - 2: Medium anomaly
        - 3: High anomaly
        """
        categories = np.zeros(len(point_residuals), dtype=int)

        categories[point_residuals >= thresholds['low']] = 1
        categories[point_residuals >= thresholds['medium']] = 2
        categories[point_residuals >= thresholds['high']] = 3

        return categories

    def get_neighbor_info(self, data: np.ndarray, point_idx: int) -> Dict:
        """
        Get detailed information about a point's neighbors.

        Useful for debugging and understanding predictions.
        """
        if self.nn_model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        data_scaled = self.scaler.transform(data)
        point_scaled = data_scaled[point_idx:point_idx+1]

        distances, indices = self.nn_model.kneighbors(point_scaled)

        return {
            'point_index': point_idx,
            'point_value': data[point_idx],
            'neighbor_indices': indices[0][1:],
            'neighbor_distances': distances[0][1:],
            'neighbor_values': self.training_data[indices[0][1:]]
        }


class MultiScaleResidualCalculator:
    """
    Multi-scale residual calculator that combines residuals
    computed at different k values for robust anomaly detection.
    """

    def __init__(self, k_values: List[int] = None):
        """
        Initialize with multiple k values.

        Args:
            k_values: List of k values for nearest neighbors.
                      Default is [3, 5, 10, 20]
        """
        self.k_values = k_values if k_values is not None else [3, 5, 10, 20]
        self.calculators: Dict[int, ResidualCalculator] = {}

    def fit(self, data: np.ndarray) -> 'MultiScaleResidualCalculator':
        """Fit calculators for all k values."""
        max_k = len(data) - 1

        for k in self.k_values:
            if k < max_k:
                calc = ResidualCalculator(n_neighbors=k)
                calc.fit(data)
                self.calculators[k] = calc

        return self

    def calculate_residuals(self, data: np.ndarray,
                            cluster_labels: Optional[np.ndarray] = None) -> ResidualResult:
        """
        Calculate multi-scale residuals.

        Combines residuals from different k values using weighted average.
        """
        if not self.calculators:
            raise ValueError("No calculators fitted. Call fit() first.")

        all_residuals = []
        all_feature_residuals = []
        weights = []

        for k, calc in self.calculators.items():
            result = calc.calculate_residuals(data, cluster_labels)
            all_residuals.append(result.point_residuals)
            all_feature_residuals.append(result.feature_residuals)
            # Higher k gets lower weight (captures broader patterns)
            weights.append(1.0 / np.sqrt(k))

        # Normalize weights
        weights = np.array(weights) / sum(weights)

        # Combine residuals
        combined_residuals = np.zeros_like(all_residuals[0])
        combined_feature_residuals = np.zeros_like(all_feature_residuals[0])

        for i, (res, feat_res) in enumerate(zip(all_residuals, all_feature_residuals)):
            combined_residuals += weights[i] * res
            combined_feature_residuals += weights[i] * feat_res

        # Use the last calculator's methods for OMR and thresholds
        last_calc = list(self.calculators.values())[-1]
        omr = last_calc._calculate_omr(combined_residuals)
        anomaly_scores = last_calc._calculate_anomaly_scores(combined_residuals)
        thresholds = last_calc._calculate_thresholds(combined_residuals)
        categories = last_calc._categorize_anomalies(combined_residuals, thresholds)

        return ResidualResult(
            point_residuals=combined_residuals,
            overall_model_residual=omr,
            feature_residuals=combined_feature_residuals,
            anomaly_scores=anomaly_scores,
            threshold_low=thresholds['low'],
            threshold_medium=thresholds['medium'],
            threshold_high=thresholds['high'],
            anomaly_categories=categories
        )
