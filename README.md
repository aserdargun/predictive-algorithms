# Clustering-Based Predictive Maintenance Tool

A Python tool for predictive maintenance using multiple clustering algorithms to detect anomalies and calculate Overall Model Residuals (OMR). Inspired by the AVEVA Predictive Analytics approach.

## Features

- **Multiple Clustering Algorithms**:
  - K-means
  - DBSCAN
  - OPTICS
  - SOM (Self-Organizing Maps)
  - LSH (Locality-Sensitive Hashing)

- **Nearest Neighbor Residual Calculation**: Uses K-NN to compute point residuals and predict expected values

- **Overall Model Residual (OMR)**: Single metric representing overall model health with categorization (Normal, Low, Medium, High)

- **Significant Factor Analysis**: Identifies which sensors/features contribute most to detected anomalies

- **Multi-scale Analysis**: Combines residuals from different k values for robust detection

- **Visualization**: Built-in plotting capabilities with matplotlib and plotly support

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.predictive_maintenance import PredictiveMaintenanceTool

# Define your sensor names (max 15)
feature_names = [
    "Temperature", "Pressure", "Vibration", "Flow_Rate", "RPM"
]

# Initialize the tool
pm_tool = PredictiveMaintenanceTool(
    feature_names=feature_names,
    n_neighbors=5,
    multi_scale=True,
    algorithms=['kmeans', 'dbscan', 'optics', 'som', 'lsh']
)

# Fit on normal operating data
pm_tool.fit(training_data)

# Analyze new data for anomalies
report = pm_tool.analyze(new_data)

# Print the report
print(pm_tool.print_report(report))

# Get specific metrics
omr = report.overall_model_residual
anomaly_rate = report.anomaly_rate
top_factors = report.top_contributing_factors
```

## How It Works

### 1. Clustering Phase

The tool applies multiple clustering algorithms to identify patterns in the data:

- **K-means**: Partitions data into k clusters based on centroid distance
- **DBSCAN**: Density-based clustering that can identify noise points
- **OPTICS**: Ordering points to identify clustering structure
- **SOM**: Neural network-based topological clustering
- **LSH**: Locality-sensitive hashing for approximate clustering

An ensemble approach combines results from all algorithms for robust cluster assignment.

### 2. Residual Calculation

For each data point, residuals are calculated as:

```
residual = actual_value - predicted_value
```

Where `predicted_value` is the weighted average of K nearest neighbors. This approach detects when current readings deviate from expected patterns based on historical data.

### 3. Overall Model Residual (OMR)

OMR is a composite metric:

```
OMR = 0.7 * mean_residual + 0.3 * outlier_penalty
```

Categories:
- **Normal**: OMR < 0.5
- **Low**: 0.5 <= OMR < 1.0
- **Medium**: 1.0 <= OMR < 2.0
- **High**: OMR >= 2.0

### 4. Factor Contribution Analysis

The tool identifies which sensors contribute most to anomalies by:
- Decomposing residuals by feature
- Normalizing by baseline statistics
- Ranking features by contribution percentage

## API Reference

### PredictiveMaintenanceTool

Main class for predictive maintenance analysis.

```python
pm_tool = PredictiveMaintenanceTool(
    feature_names=None,      # List of sensor names (max 15)
    n_neighbors=5,           # K for nearest neighbor
    multi_scale=True,        # Use multiple k values
    algorithms=None,         # List of clustering algorithms
    random_state=42          # Random seed
)
```

**Methods:**

- `fit(training_data)`: Train on normal operating data
- `analyze(data)`: Analyze data and return MaintenanceReport
- `get_anomaly_scores(data)`: Get anomaly scores (0-100)
- `get_residuals(data)`: Get point residuals and OMR
- `get_cluster_labels(algorithm=None)`: Get cluster assignments

### MaintenanceReport

Contains analysis results:

- `overall_model_residual`: OMR value
- `omr_category`: 'Normal', 'Low', 'Medium', or 'High'
- `anomaly_rate`: Percentage of anomalous points
- `clustering_summary`: Results from each algorithm
- `factor_contributions`: Feature contribution analysis
- `anomaly_diagnostics`: Detailed diagnostics for anomaly points

### MaintenanceVisualizer

Visualization tools for the analysis.

```python
viz = MaintenanceVisualizer(backend='matplotlib')  # or 'plotly'

# Available plots
viz.plot_overall_residuals(residuals, thresholds)
viz.plot_anomaly_categories(categories)
viz.plot_factor_contributions(names, contributions)
viz.plot_cluster_comparison(data, cluster_results)
viz.plot_omr_gauge(omr, category)
viz.plot_timeseries_with_anomalies(data, names, categories)
viz.create_dashboard(report, data)  # plotly only
```

## Example Output

```
============================================================
PREDICTIVE MAINTENANCE ANALYSIS REPORT
============================================================
Timestamp: 2024-01-15T10:30:00
Samples analyzed: 150
Features: 10

----------------------------------------
OVERALL STATUS
----------------------------------------
Overall Model Residual (OMR): 0.8542
OMR Category: Low
Anomaly Rate: 8.0%

----------------------------------------
TOP CONTRIBUTING FACTORS
----------------------------------------
  Vibration: 28.5%
  Temperature: 22.3%
  Pressure: 15.7%
  RPM: 12.1%
  Flow_Rate: 8.4%

----------------------------------------
ANOMALY DIAGNOSTICS
----------------------------------------
Point 45 (Medium anomaly):
  Score: 2.3456
  Summary: Vibration: 4.52 (+2.8 sigma above normal); Temperature: 85.3 (+1.9 sigma above normal)
  Recommendations:
    - Schedule inspection of related components.
    - Check Vibration: value is significantly above normal range.
============================================================
```

## References

- [AVEVA Predictive Analytics](https://www.aveva.com/en/products/predictive-analytics/)
- [scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [DBSCAN Algorithm](https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html)

## License

MIT License
