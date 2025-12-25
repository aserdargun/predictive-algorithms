"""
Example Usage of the Predictive Maintenance Tool

This script demonstrates how to use the clustering-based
predictive maintenance tool with synthetic sensor data.
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from src.predictive_maintenance import PredictiveMaintenanceTool, load_timeseries_data
from src.visualization import MaintenanceVisualizer


def generate_synthetic_data(n_samples: int = 500,
                            n_features: int = 10,
                            anomaly_ratio: float = 0.05,
                            random_state: int = 42) -> tuple:
    """
    Generate synthetic sensor data with injected anomalies.

    Returns:
        Tuple of (training_data, test_data, test_labels)
    """
    np.random.seed(random_state)

    # Generate normal operating data
    # Simulate correlated sensor readings
    n_train = int(n_samples * 0.7)
    n_test = n_samples - n_train

    # Base signals
    base = np.random.randn(n_samples, 3)

    # Create correlated features
    data = np.zeros((n_samples, n_features))

    # First few features are base signals with noise
    for i in range(min(3, n_features)):
        data[:, i] = base[:, i] + np.random.randn(n_samples) * 0.1

    # Remaining features are combinations of base signals
    for i in range(3, n_features):
        weights = np.random.randn(3)
        data[:, i] = np.dot(base, weights) + np.random.randn(n_samples) * 0.2

    # Normalize
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    # Split into train/test
    train_data = data[:n_train]
    test_data = data[n_train:].copy()

    # Inject anomalies into test data
    n_anomalies = int(n_test * anomaly_ratio)
    anomaly_indices = np.random.choice(n_test, n_anomalies, replace=False)

    # Create labels
    test_labels = np.zeros(n_test, dtype=int)

    for idx in anomaly_indices:
        # Random anomaly type
        anomaly_type = np.random.choice(['spike', 'drift', 'correlation_break'])

        if anomaly_type == 'spike':
            # Sudden spike in random features
            affected_features = np.random.choice(
                n_features, np.random.randint(1, 4), replace=False
            )
            for feat in affected_features:
                test_data[idx, feat] += np.random.choice([-1, 1]) * np.random.uniform(3, 5)
            test_labels[idx] = 3  # High anomaly

        elif anomaly_type == 'drift':
            # Gradual drift over a window
            window = min(10, n_test - idx)
            affected_feat = np.random.randint(n_features)
            drift = np.linspace(0, np.random.uniform(2, 3), window)
            test_data[idx:idx+window, affected_feat] += drift
            test_labels[idx:idx+window] = np.maximum(test_labels[idx:idx+window], 1)

        else:  # correlation_break
            # Break correlation between features
            affected_features = np.random.choice(n_features, 2, replace=False)
            test_data[idx, affected_features[0]] += 2
            test_data[idx, affected_features[1]] -= 2
            test_labels[idx] = 2  # Medium anomaly

    return train_data, test_data, test_labels


def main():
    """Main example function."""
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE TOOL - EXAMPLE USAGE")
    print("=" * 60)

    # Generate synthetic data
    print("\n1. Generating synthetic sensor data...")
    feature_names = [
        "Temperature", "Pressure", "Vibration", "Flow_Rate",
        "RPM", "Current", "Voltage", "Oil_Level",
        "Humidity", "Ambient_Temp"
    ]

    train_data, test_data, true_labels = generate_synthetic_data(
        n_samples=500,
        n_features=10,
        anomaly_ratio=0.08
    )

    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    print(f"   Features: {len(feature_names)}")

    # Initialize the tool
    print("\n2. Initializing Predictive Maintenance Tool...")
    pm_tool = PredictiveMaintenanceTool(
        feature_names=feature_names,
        n_neighbors=5,
        multi_scale=True,
        algorithms=['kmeans', 'dbscan', 'optics', 'som', 'lsh'],
        random_state=42
    )

    # Fit on training data (normal operation)
    print("\n3. Training model on normal operating data...")
    pm_tool.fit(train_data)
    print("   Model training complete!")

    # Analyze test data
    print("\n4. Analyzing test data for anomalies...")
    report = pm_tool.analyze(test_data, detailed_diagnostics=True)

    # Print the report
    print("\n" + pm_tool.print_report(report))

    # Get individual metrics
    print("\n5. Additional Analysis:")
    print("-" * 40)

    # Anomaly scores
    scores = pm_tool.get_anomaly_scores(test_data)
    print(f"   Mean anomaly score: {np.mean(scores):.2f}")
    print(f"   Max anomaly score: {np.max(scores):.2f}")

    # Residuals
    residuals, omr = pm_tool.get_residuals(test_data)
    print(f"   Mean residual: {np.mean(residuals):.4f}")
    print(f"   Overall Model Residual: {omr:.4f}")

    # Cluster labels
    labels = pm_tool.get_cluster_labels()
    print(f"   Ensemble clusters: {len(set(labels))}")

    # Compare with true labels
    print("\n6. Comparison with True Anomaly Labels:")
    print("-" * 40)
    detected_anomalies = np.where(report.anomaly_categories > 0)[0]
    true_anomalies = np.where(true_labels > 0)[0]

    if len(true_anomalies) > 0:
        # Calculate detection metrics
        true_positives = len(set(detected_anomalies) & set(true_anomalies))
        precision = true_positives / len(detected_anomalies) if len(detected_anomalies) > 0 else 0
        recall = true_positives / len(true_anomalies)

        print(f"   True anomalies: {len(true_anomalies)}")
        print(f"   Detected anomalies: {len(detected_anomalies)}")
        print(f"   True positives: {true_positives}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")

    # Export report
    print("\n7. Exporting Report...")
    report_json = report.to_json()
    with open('maintenance_report.json', 'w') as f:
        f.write(report_json)
    print("   Report exported to 'maintenance_report.json'")

    # Visualization (if available)
    try:
        print("\n8. Creating Visualizations...")
        viz = MaintenanceVisualizer(backend='matplotlib')

        # Create and save plots
        thresholds = {
            'low': report.residual_summary['threshold_low'],
            'medium': report.residual_summary['threshold_medium'],
            'high': report.residual_summary['threshold_high']
        }

        fig1 = viz.plot_overall_residuals(residuals, thresholds)
        fig1.savefig('residuals_plot.png', dpi=150, bbox_inches='tight')
        print("   Saved: residuals_plot.png")

        fig2 = viz.plot_anomaly_categories(report.anomaly_categories)
        fig2.savefig('anomaly_categories.png', dpi=150, bbox_inches='tight')
        print("   Saved: anomaly_categories.png")

        fig3 = viz.plot_factor_contributions(
            report.factor_contributions.feature_names,
            report.factor_contributions.contribution_percentages
        )
        fig3.savefig('factor_contributions.png', dpi=150, bbox_inches='tight')
        print("   Saved: factor_contributions.png")

        fig4 = viz.plot_omr_gauge(report.overall_model_residual, report.omr_category)
        fig4.savefig('omr_gauge.png', dpi=150, bbox_inches='tight')
        print("   Saved: omr_gauge.png")

        fig5 = viz.plot_timeseries_with_anomalies(
            test_data, feature_names, report.anomaly_categories
        )
        fig5.savefig('timeseries_anomalies.png', dpi=150, bbox_inches='tight')
        print("   Saved: timeseries_anomalies.png")

        # Cluster comparison (use training data since cluster labels are from training)
        fig6 = viz.plot_cluster_comparison(
            train_data,
            pm_tool.clustering_engine.get_all_results(),
            feature_x=0, feature_y=1
        )
        fig6.savefig('cluster_comparison.png', dpi=150, bbox_inches='tight')
        print("   Saved: cluster_comparison.png")

        import matplotlib.pyplot as plt
        plt.close('all')

    except ImportError as e:
        print(f"   Visualization skipped (missing dependency: {e})")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
