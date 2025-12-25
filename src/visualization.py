"""
Visualization Module for Predictive Maintenance

Provides plotting functions for residuals, clusters,
anomalies, and factor contributions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not installed. Visualization functions will not work.")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


class MaintenanceVisualizer:
    """
    Visualization tools for predictive maintenance analysis.

    Supports both matplotlib (static) and plotly (interactive) backends.
    """

    def __init__(self, backend: str = 'matplotlib', style: str = 'darkgrid'):
        """
        Initialize visualizer.

        Args:
            backend: 'matplotlib' or 'plotly'
            style: Seaborn style for matplotlib plots
        """
        self.backend = backend

        if backend == 'matplotlib' and not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for this backend")
        if backend == 'plotly' and not HAS_PLOTLY:
            raise ImportError("plotly is required for this backend")

        if HAS_SEABORN and backend == 'matplotlib':
            sns.set_style(style)

        # Color scheme for anomaly categories
        self.anomaly_colors = {
            0: '#2ecc71',  # Normal - green
            1: '#f1c40f',  # Low - yellow
            2: '#e67e22',  # Medium - orange
            3: '#e74c3c'   # High - red
        }

    def plot_overall_residuals(self, residuals: np.ndarray,
                                thresholds: Optional[Dict] = None,
                                title: str = "Overall Model Residuals",
                                figsize: Tuple[int, int] = (12, 6)):
        """
        Plot residuals over time with threshold lines.
        """
        if self.backend == 'matplotlib':
            return self._plot_residuals_mpl(residuals, thresholds, title, figsize)
        else:
            return self._plot_residuals_plotly(residuals, thresholds, title)

    def _plot_residuals_mpl(self, residuals, thresholds, title, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(residuals))
        ax.plot(x, residuals, 'b-', linewidth=0.8, alpha=0.7, label='Residual')
        ax.scatter(x, residuals, c='blue', s=10, alpha=0.5)

        if thresholds:
            ax.axhline(y=thresholds.get('low', 0), color='#f1c40f',
                      linestyle='--', label='Low Threshold', linewidth=1.5)
            ax.axhline(y=thresholds.get('medium', 0), color='#e67e22',
                      linestyle='--', label='Medium Threshold', linewidth=1.5)
            ax.axhline(y=thresholds.get('high', 0), color='#e74c3c',
                      linestyle='--', label='High Threshold', linewidth=1.5)

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Residual Value')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _plot_residuals_plotly(self, residuals, thresholds, title):
        fig = go.Figure()

        x = list(range(len(residuals)))

        fig.add_trace(go.Scatter(
            x=x, y=residuals,
            mode='lines+markers',
            name='Residual',
            line=dict(color='blue', width=1),
            marker=dict(size=4)
        ))

        if thresholds:
            for level, color in [('low', '#f1c40f'), ('medium', '#e67e22'), ('high', '#e74c3c')]:
                if level in thresholds:
                    fig.add_hline(
                        y=thresholds[level],
                        line=dict(color=color, dash='dash'),
                        annotation_text=f"{level.capitalize()} Threshold"
                    )

        fig.update_layout(
            title=title,
            xaxis_title='Sample Index',
            yaxis_title='Residual Value',
            template='plotly_white'
        )

        return fig

    def plot_anomaly_categories(self, categories: np.ndarray,
                                 figsize: Tuple[int, int] = (12, 4)):
        """
        Plot anomaly categories as a heatmap strip.
        """
        if self.backend == 'matplotlib':
            return self._plot_categories_mpl(categories, figsize)
        else:
            return self._plot_categories_plotly(categories)

    def _plot_categories_mpl(self, categories, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        colors = [self.anomaly_colors[c] for c in categories]
        ax.bar(range(len(categories)), np.ones(len(categories)),
               color=colors, width=1.0, edgecolor='none')

        ax.set_xlim(0, len(categories))
        ax.set_ylim(0, 1)
        ax.set_xlabel('Sample Index')
        ax.set_title('Anomaly Categories')
        ax.set_yticks([])

        # Legend
        patches = [
            mpatches.Patch(color=self.anomaly_colors[0], label='Normal'),
            mpatches.Patch(color=self.anomaly_colors[1], label='Low'),
            mpatches.Patch(color=self.anomaly_colors[2], label='Medium'),
            mpatches.Patch(color=self.anomaly_colors[3], label='High')
        ]
        ax.legend(handles=patches, loc='upper right', ncol=4)

        plt.tight_layout()
        return fig

    def _plot_categories_plotly(self, categories):
        category_names = ['Normal', 'Low', 'Medium', 'High']
        colors = [self.anomaly_colors[c] for c in categories]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=list(range(len(categories))),
            y=[1] * len(categories),
            marker_color=colors,
            hovertext=[category_names[c] for c in categories],
            hoverinfo='text+x'
        ))

        fig.update_layout(
            title='Anomaly Categories',
            xaxis_title='Sample Index',
            showlegend=False,
            yaxis=dict(visible=False),
            template='plotly_white'
        )

        return fig

    def plot_factor_contributions(self, feature_names: List[str],
                                   contributions: np.ndarray,
                                   title: str = "Factor Contributions",
                                   figsize: Tuple[int, int] = (10, 6)):
        """
        Plot factor contribution as horizontal bar chart.
        """
        if self.backend == 'matplotlib':
            return self._plot_factors_mpl(feature_names, contributions, title, figsize)
        else:
            return self._plot_factors_plotly(feature_names, contributions, title)

    def _plot_factors_mpl(self, feature_names, contributions, title, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        # Sort by contribution
        sorted_idx = np.argsort(contributions)
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_contribs = contributions[sorted_idx]

        # Color gradient based on contribution
        colors = plt.cm.RdYlGn_r(sorted_contribs / max(sorted_contribs) if max(sorted_contribs) > 0 else sorted_contribs)

        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_contribs, color=colors, edgecolor='gray', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Contribution (%)')
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)

        # Add percentage labels
        for i, (name, contrib) in enumerate(zip(sorted_names, sorted_contribs)):
            ax.text(contrib + 0.5, i, f'{contrib:.1f}%', va='center', fontsize=9)

        plt.tight_layout()
        return fig

    def _plot_factors_plotly(self, feature_names, contributions, title):
        sorted_idx = np.argsort(contributions)
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_contribs = contributions[sorted_idx]

        fig = go.Figure(go.Bar(
            x=sorted_contribs,
            y=sorted_names,
            orientation='h',
            marker=dict(
                color=sorted_contribs,
                colorscale='RdYlGn_r',
                showscale=True
            ),
            text=[f'{c:.1f}%' for c in sorted_contribs],
            textposition='outside'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Contribution (%)',
            template='plotly_white'
        )

        return fig

    def plot_cluster_comparison(self, data: np.ndarray,
                                 cluster_results: Dict,
                                 feature_x: int = 0,
                                 feature_y: int = 1,
                                 figsize: Tuple[int, int] = (15, 10)):
        """
        Plot clustering results from different algorithms side by side.
        """
        if self.backend == 'matplotlib':
            return self._plot_clusters_mpl(data, cluster_results, feature_x, feature_y, figsize)
        else:
            return self._plot_clusters_plotly(data, cluster_results, feature_x, feature_y)

    def _plot_clusters_mpl(self, data, cluster_results, feature_x, feature_y, figsize):
        n_algos = len(cluster_results)
        n_cols = min(3, n_algos)
        n_rows = (n_algos + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]

        for idx, (alg_name, result) in enumerate(cluster_results.items()):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row][col]

            labels = result.labels
            unique_labels = set(labels)

            for label in unique_labels:
                mask = labels == label
                if label == -1:
                    ax.scatter(data[mask, feature_x], data[mask, feature_y],
                              c='gray', s=20, alpha=0.5, label='Noise')
                else:
                    ax.scatter(data[mask, feature_x], data[mask, feature_y],
                              s=30, alpha=0.7, label=f'Cluster {label}')

            ax.set_title(f'{alg_name.upper()}\n({result.n_clusters} clusters)')
            ax.set_xlabel(f'Feature {feature_x}')
            ax.set_ylabel(f'Feature {feature_y}')
            ax.legend(loc='best', fontsize=8)

        # Hide empty subplots
        for idx in range(n_algos, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row][col].axis('off')

        plt.tight_layout()
        return fig

    def _plot_clusters_plotly(self, data, cluster_results, feature_x, feature_y):
        n_algos = len(cluster_results)
        n_cols = min(3, n_algos)
        n_rows = (n_algos + n_cols - 1) // n_cols

        fig = make_subplots(rows=n_rows, cols=n_cols,
                           subplot_titles=[f"{alg.upper()} ({r.n_clusters} clusters)"
                                          for alg, r in cluster_results.items()])

        for idx, (alg_name, result) in enumerate(cluster_results.items()):
            row, col = idx // n_cols + 1, idx % n_cols + 1
            labels = result.labels

            fig.add_trace(
                go.Scatter(
                    x=data[:, feature_x],
                    y=data[:, feature_y],
                    mode='markers',
                    marker=dict(
                        color=labels,
                        colorscale='Viridis',
                        size=8,
                        opacity=0.7
                    ),
                    name=alg_name
                ),
                row=row, col=col
            )

        fig.update_layout(
            title='Clustering Comparison',
            template='plotly_white',
            showlegend=False
        )

        return fig

    def plot_omr_gauge(self, omr: float, category: str,
                        figsize: Tuple[int, int] = (6, 4)):
        """
        Plot OMR as a gauge/dial chart.
        """
        if self.backend == 'plotly':
            return self._plot_omr_gauge_plotly(omr, category)
        else:
            return self._plot_omr_gauge_mpl(omr, category, figsize)

    def _plot_omr_gauge_mpl(self, omr, category, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        # Create a simple bar representation
        colors = {'Normal': '#2ecc71', 'Low': '#f1c40f',
                  'Medium': '#e67e22', 'High': '#e74c3c'}
        color = colors.get(category, 'gray')

        ax.barh([0], [omr], color=color, height=0.5)
        ax.barh([0], [max(3, omr * 1.2) - omr], left=omr,
               color='lightgray', height=0.5)

        ax.set_xlim(0, max(3, omr * 1.2))
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xlabel('OMR Value')
        ax.set_title(f'Overall Model Residual: {omr:.4f}\nCategory: {category}')

        # Add threshold markers
        for thresh, label in [(0.5, 'Low'), (1.0, 'Med'), (2.0, 'High')]:
            if thresh <= max(3, omr * 1.2):
                ax.axvline(x=thresh, color='gray', linestyle=':', alpha=0.7)
                ax.text(thresh, 0.35, label, ha='center', fontsize=8)

        plt.tight_layout()
        return fig

    def _plot_omr_gauge_plotly(self, omr, category):
        colors = {'Normal': 'green', 'Low': 'yellow',
                  'Medium': 'orange', 'High': 'red'}

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=omr,
            title={'text': f"Overall Model Residual ({category})"},
            gauge={
                'axis': {'range': [0, max(3, omr * 1.2)]},
                'bar': {'color': colors.get(category, 'gray')},
                'steps': [
                    {'range': [0, 0.5], 'color': '#d4edda'},
                    {'range': [0.5, 1.0], 'color': '#fff3cd'},
                    {'range': [1.0, 2.0], 'color': '#ffe4c4'},
                    {'range': [2.0, max(3, omr * 1.2)], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': omr
                }
            }
        ))

        fig.update_layout(template='plotly_white')
        return fig

    def plot_timeseries_with_anomalies(self, data: np.ndarray,
                                        feature_names: List[str],
                                        anomaly_categories: np.ndarray,
                                        figsize: Tuple[int, int] = (14, 10)):
        """
        Plot all time series with anomaly highlighting.
        """
        if self.backend == 'matplotlib':
            return self._plot_ts_anomalies_mpl(data, feature_names, anomaly_categories, figsize)
        else:
            return self._plot_ts_anomalies_plotly(data, feature_names, anomaly_categories)

    def _plot_ts_anomalies_mpl(self, data, feature_names, anomaly_categories, figsize):
        n_features = min(data.shape[1], len(feature_names))
        fig, axes = plt.subplots(n_features, 1, figsize=figsize, sharex=True)

        if n_features == 1:
            axes = [axes]

        x = np.arange(len(data))

        for i, (ax, name) in enumerate(zip(axes, feature_names[:n_features])):
            ax.plot(x, data[:, i], 'b-', linewidth=0.8, alpha=0.7)

            # Highlight anomalies
            for cat in [1, 2, 3]:
                mask = anomaly_categories == cat
                if mask.any():
                    ax.scatter(x[mask], data[mask, i],
                              c=self.anomaly_colors[cat], s=30, zorder=5,
                              label=['', 'Low', 'Medium', 'High'][cat])

            ax.set_ylabel(name, fontsize=9)
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Sample Index')
        plt.suptitle('Time Series with Anomalies Highlighted', y=1.02)
        plt.tight_layout()
        return fig

    def _plot_ts_anomalies_plotly(self, data, feature_names, anomaly_categories):
        n_features = min(data.shape[1], len(feature_names))

        fig = make_subplots(rows=n_features, cols=1,
                           shared_xaxes=True,
                           subplot_titles=feature_names[:n_features])

        x = list(range(len(data)))

        for i in range(n_features):
            # Main line
            fig.add_trace(
                go.Scatter(x=x, y=data[:, i], mode='lines',
                          name=feature_names[i], line=dict(color='blue', width=1)),
                row=i+1, col=1
            )

            # Anomaly points
            for cat, cat_name in [(1, 'Low'), (2, 'Medium'), (3, 'High')]:
                mask = anomaly_categories == cat
                if mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=[x[j] for j in np.where(mask)[0]],
                            y=data[mask, i],
                            mode='markers',
                            name=f'{cat_name} Anomaly',
                            marker=dict(color=self.anomaly_colors[cat], size=8),
                            showlegend=(i == 0)
                        ),
                        row=i+1, col=1
                    )

        fig.update_layout(
            title='Time Series with Anomalies Highlighted',
            template='plotly_white',
            height=200 * n_features
        )

        return fig

    def create_dashboard(self, report, data: np.ndarray):
        """
        Create a comprehensive dashboard with all visualizations.

        Only available with plotly backend.
        """
        if self.backend != 'plotly':
            raise ValueError("Dashboard requires plotly backend")

        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "scatter", "colspan": 2}, None],
                [{"type": "bar"}, {"type": "scatter"}]
            ],
            subplot_titles=[
                'Overall Model Residual', 'Anomaly Distribution',
                'Residuals Over Time',
                'Factor Contributions', 'Cluster View'
            ]
        )

        # OMR Gauge
        colors = {'Normal': 'green', 'Low': 'yellow',
                  'Medium': 'orange', 'High': 'red'}
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=report.overall_model_residual,
                gauge={
                    'axis': {'range': [0, 3]},
                    'bar': {'color': colors.get(report.omr_category, 'gray')},
                    'steps': [
                        {'range': [0, 0.5], 'color': '#d4edda'},
                        {'range': [0.5, 1.0], 'color': '#fff3cd'},
                        {'range': [1.0, 2.0], 'color': '#ffe4c4'},
                        {'range': [2.0, 3], 'color': '#f8d7da'}
                    ]
                }
            ),
            row=1, col=1
        )

        # Anomaly Distribution
        anomaly_counts = [
            np.sum(report.anomaly_categories == 0),
            np.sum(report.anomaly_categories == 1),
            np.sum(report.anomaly_categories == 2),
            np.sum(report.anomaly_categories == 3)
        ]
        fig.add_trace(
            go.Bar(
                x=['Normal', 'Low', 'Medium', 'High'],
                y=anomaly_counts,
                marker_color=['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            ),
            row=1, col=2
        )

        # Residuals over time - need to recalculate or pass residuals
        # For now, use anomaly scores as proxy
        fig.add_trace(
            go.Scatter(
                x=list(range(len(report.anomaly_categories))),
                y=[self.anomaly_colors[c] for c in report.anomaly_categories],
                mode='markers',
                marker=dict(
                    color=[self.anomaly_colors[c] for c in report.anomaly_categories],
                    size=8
                )
            ),
            row=2, col=1
        )

        # Factor contributions
        contribs = report.factor_contributions.contribution_percentages
        names = report.factor_contributions.feature_names
        sorted_idx = np.argsort(contribs)[-10:]  # Top 10
        fig.add_trace(
            go.Bar(
                x=[contribs[i] for i in sorted_idx],
                y=[names[i] for i in sorted_idx],
                orientation='h',
                marker=dict(color='steelblue')
            ),
            row=3, col=1
        )

        # Cluster scatter
        if data.shape[1] >= 2:
            fig.add_trace(
                go.Scatter(
                    x=data[:, 0],
                    y=data[:, 1],
                    mode='markers',
                    marker=dict(
                        color=report.anomaly_categories,
                        colorscale=[[0, '#2ecc71'], [0.33, '#f1c40f'],
                                   [0.66, '#e67e22'], [1, '#e74c3c']],
                        size=8
                    )
                ),
                row=3, col=2
            )

        fig.update_layout(
            title=f'Predictive Maintenance Dashboard - {report.timestamp}',
            template='plotly_white',
            height=900,
            showlegend=False
        )

        return fig
