"""
Tests for the Predictive Maintenance Tool.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, '..')

from src.clustering import ClusteringEngine, LSHIndex, SOMClustering
from src.residual_calculator import ResidualCalculator, MultiScaleResidualCalculator
from src.factor_analyzer import SignificantFactorAnalyzer
from src.predictive_maintenance import PredictiveMaintenanceTool


class TestClusteringEngine:
    """Tests for the ClusteringEngine class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        # Create 3 clear clusters
        cluster1 = np.random.randn(30, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(30, 5) + np.array([5, 5, 5, 5, 5])
        cluster3 = np.random.randn(30, 5) + np.array([10, 0, 10, 0, 10])
        return np.vstack([cluster1, cluster2, cluster3])

    def test_kmeans_clustering(self, sample_data):
        """Test K-means clustering."""
        engine = ClusteringEngine(random_state=42)
        engine.fit(sample_data, algorithms=['kmeans'], params={'kmeans': {'n_clusters': 3}})

        result = engine.get_result('kmeans')
        assert result.n_clusters == 3
        assert len(result.labels) == len(sample_data)
        assert result.cluster_centers.shape == (3, 5)

    def test_dbscan_clustering(self, sample_data):
        """Test DBSCAN clustering."""
        engine = ClusteringEngine(random_state=42)
        engine.fit(sample_data, algorithms=['dbscan'])

        result = engine.get_result('dbscan')
        assert result.n_clusters >= 1
        assert len(result.labels) == len(sample_data)

    def test_optics_clustering(self, sample_data):
        """Test OPTICS clustering."""
        engine = ClusteringEngine(random_state=42)
        engine.fit(sample_data, algorithms=['optics'])

        result = engine.get_result('optics')
        assert len(result.labels) == len(sample_data)

    def test_som_clustering(self, sample_data):
        """Test SOM clustering."""
        engine = ClusteringEngine(random_state=42)
        engine.fit(sample_data, algorithms=['som'],
                  params={'som': {'grid_size': (5, 5), 'n_iterations': 100}})

        result = engine.get_result('som')
        assert len(result.labels) == len(sample_data)

    def test_lsh_clustering(self, sample_data):
        """Test LSH clustering."""
        engine = ClusteringEngine(random_state=42)
        engine.fit(sample_data, algorithms=['lsh'])

        result = engine.get_result('lsh')
        assert len(result.labels) == len(sample_data)

    def test_ensemble_labels(self, sample_data):
        """Test ensemble clustering."""
        engine = ClusteringEngine(random_state=42)
        engine.fit(sample_data)

        ensemble_labels = engine.get_ensemble_labels()
        assert len(ensemble_labels) == len(sample_data)

    def test_all_algorithms(self, sample_data):
        """Test running all algorithms together."""
        engine = ClusteringEngine(random_state=42)
        engine.fit(sample_data)

        results = engine.get_all_results()
        assert 'kmeans' in results
        assert 'dbscan' in results
        assert 'optics' in results
        assert 'som' in results
        assert 'lsh' in results


class TestLSHIndex:
    """Tests for the LSH Index class."""

    def test_fit_and_query(self):
        """Test LSH fit and query."""
        np.random.seed(42)
        data = np.random.randn(100, 10)

        lsh = LSHIndex(n_hash_tables=5, n_hash_functions=4, random_state=42)
        lsh.fit(data)

        # Query a known point
        distances, indices = lsh.query(data[0], k=5)

        assert len(indices) <= 5
        assert 0 in indices  # The point itself should be found

    def test_get_clusters(self):
        """Test LSH cluster assignment."""
        np.random.seed(42)
        data = np.random.randn(50, 5)

        lsh = LSHIndex(n_hash_tables=10, n_hash_functions=8, random_state=42)
        lsh.fit(data)
        labels = lsh.get_clusters()

        assert len(labels) == len(data)


class TestSOMClustering:
    """Tests for the SOM clustering class."""

    def test_fit_and_predict(self):
        """Test SOM fit and predict."""
        np.random.seed(42)
        data = np.random.randn(100, 5)

        som = SOMClustering(grid_size=(5, 5), n_iterations=100, random_state=42)
        som.fit(data)
        labels = som.predict(data)

        assert len(labels) == len(data)
        assert som.weights is not None

    def test_cluster_centers(self):
        """Test SOM cluster centers."""
        np.random.seed(42)
        data = np.random.randn(100, 5)

        som = SOMClustering(grid_size=(3, 3), n_iterations=100, random_state=42)
        som.fit(data)
        centers = som.get_cluster_centers()

        assert centers.shape == (9, 5)  # 3x3 grid = 9 nodes


class TestResidualCalculator:
    """Tests for the ResidualCalculator class."""

    @pytest.fixture
    def training_data(self):
        """Generate training data."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    def test_fit_and_calculate(self, training_data):
        """Test residual calculation."""
        calc = ResidualCalculator(n_neighbors=5)
        calc.fit(training_data)

        # Calculate residuals on same data (should be low)
        result = calc.calculate_residuals(training_data)

        assert len(result.point_residuals) == len(training_data)
        assert result.overall_model_residual >= 0
        assert len(result.anomaly_categories) == len(training_data)

    def test_anomaly_detection(self, training_data):
        """Test that anomalies are detected."""
        calc = ResidualCalculator(n_neighbors=5)
        calc.fit(training_data)

        # Create test data with anomalies
        test_data = training_data.copy()
        test_data[0] += 10  # Inject anomaly

        result = calc.calculate_residuals(test_data)

        # The anomalous point should have higher residual
        assert result.point_residuals[0] > np.median(result.point_residuals)

    def test_thresholds(self, training_data):
        """Test threshold calculation."""
        calc = ResidualCalculator(n_neighbors=5)
        calc.fit(training_data)

        result = calc.calculate_residuals(training_data)

        assert result.threshold_low < result.threshold_medium
        assert result.threshold_medium < result.threshold_high


class TestMultiScaleResidualCalculator:
    """Tests for the MultiScaleResidualCalculator class."""

    def test_multi_scale(self):
        """Test multi-scale residual calculation."""
        np.random.seed(42)
        data = np.random.randn(100, 5)

        calc = MultiScaleResidualCalculator(k_values=[3, 5, 10])
        calc.fit(data)

        result = calc.calculate_residuals(data)

        assert len(result.point_residuals) == len(data)
        assert result.overall_model_residual >= 0


class TestSignificantFactorAnalyzer:
    """Tests for the SignificantFactorAnalyzer class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        return np.random.randn(100, 5)

    def test_fit_baseline(self, sample_data):
        """Test baseline fitting."""
        analyzer = SignificantFactorAnalyzer(
            feature_names=['A', 'B', 'C', 'D', 'E']
        )
        analyzer.fit_baseline(sample_data)

        assert analyzer.baseline_stats is not None
        assert 'mean' in analyzer.baseline_stats
        assert 'std' in analyzer.baseline_stats

    def test_analyze_contributions(self, sample_data):
        """Test contribution analysis."""
        analyzer = SignificantFactorAnalyzer(
            feature_names=['A', 'B', 'C', 'D', 'E']
        )
        analyzer.fit_baseline(sample_data)

        # Create fake residuals
        feature_residuals = np.random.randn(100, 5)
        point_residuals = np.linalg.norm(feature_residuals, axis=1)

        result = analyzer.analyze_contributions(
            sample_data, feature_residuals, point_residuals
        )

        assert len(result.contribution_percentages) == 5
        assert abs(sum(result.contribution_percentages) - 100) < 1e-6

    def test_analyze_point(self, sample_data):
        """Test single point analysis."""
        analyzer = SignificantFactorAnalyzer(
            feature_names=['A', 'B', 'C', 'D', 'E']
        )
        analyzer.fit_baseline(sample_data)

        feature_residuals = np.random.randn(100, 5)
        point_residuals = np.linalg.norm(feature_residuals, axis=1)
        categories = np.zeros(100, dtype=int)
        categories[0] = 2  # Medium anomaly

        diag = analyzer.analyze_point(
            0, sample_data, feature_residuals, point_residuals, categories
        )

        assert diag.point_index == 0
        assert diag.category == 'Medium'
        assert len(diag.recommended_actions) > 0


class TestPredictiveMaintenanceTool:
    """Tests for the main PredictiveMaintenanceTool class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data with clusters."""
        np.random.seed(42)
        cluster1 = np.random.randn(50, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(50, 5) + np.array([3, 3, 3, 3, 3])
        return np.vstack([cluster1, cluster2])

    def test_full_pipeline(self, sample_data):
        """Test the complete analysis pipeline."""
        feature_names = ['Temp', 'Pressure', 'Vibration', 'Flow', 'RPM']

        pm_tool = PredictiveMaintenanceTool(
            feature_names=feature_names,
            n_neighbors=5,
            multi_scale=True,
            algorithms=['kmeans', 'dbscan'],
            random_state=42
        )

        # Fit on training data
        pm_tool.fit(sample_data[:80])

        # Analyze test data
        report = pm_tool.analyze(sample_data[80:])

        assert report.n_samples == 20
        assert report.n_features == 5
        assert report.overall_model_residual >= 0
        assert report.omr_category in ['Normal', 'Low', 'Medium', 'High']
        assert 0 <= report.anomaly_rate <= 1

    def test_anomaly_scores(self, sample_data):
        """Test anomaly score calculation."""
        pm_tool = PredictiveMaintenanceTool(random_state=42)
        pm_tool.fit(sample_data[:80])

        scores = pm_tool.get_anomaly_scores(sample_data[80:])

        assert len(scores) == 20
        assert all(0 <= s <= 100 for s in scores)

    def test_residuals(self, sample_data):
        """Test residual calculation."""
        pm_tool = PredictiveMaintenanceTool(random_state=42)
        pm_tool.fit(sample_data[:80])

        residuals, omr = pm_tool.get_residuals(sample_data[80:])

        assert len(residuals) == 20
        assert omr >= 0

    def test_cluster_labels(self, sample_data):
        """Test cluster label retrieval."""
        pm_tool = PredictiveMaintenanceTool(random_state=42)
        pm_tool.fit(sample_data)

        # Ensemble labels
        labels = pm_tool.get_cluster_labels()
        assert len(labels) == len(sample_data)

        # Specific algorithm labels
        kmeans_labels = pm_tool.get_cluster_labels('kmeans')
        assert len(kmeans_labels) == len(sample_data)

    def test_max_features_limit(self):
        """Test that more than 15 features triggers warning."""
        np.random.seed(42)
        data = np.random.randn(50, 20)  # 20 features

        pm_tool = PredictiveMaintenanceTool(random_state=42)

        with pytest.warns(UserWarning, match="max is 15"):
            pm_tool.fit(data)

    def test_report_export(self, sample_data):
        """Test report export functionality."""
        pm_tool = PredictiveMaintenanceTool(random_state=42)
        pm_tool.fit(sample_data[:80])
        report = pm_tool.analyze(sample_data[80:])

        # Test to_dict
        report_dict = report.to_dict()
        assert 'overall_model_residual' in report_dict
        assert 'clustering_summary' in report_dict

        # Test to_json
        report_json = report.to_json()
        assert isinstance(report_json, str)
        assert 'overall_model_residual' in report_json

    def test_print_report(self, sample_data):
        """Test report printing."""
        pm_tool = PredictiveMaintenanceTool(random_state=42)
        pm_tool.fit(sample_data[:80])
        report = pm_tool.analyze(sample_data[80:])

        report_str = pm_tool.print_report(report)

        assert 'PREDICTIVE MAINTENANCE ANALYSIS REPORT' in report_str
        assert 'Overall Model Residual' in report_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
