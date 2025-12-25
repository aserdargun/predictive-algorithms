"""
Clustering Engine Module

Implements multiple clustering algorithms for predictive maintenance:
- K-means
- DBSCAN
- OPTICS
- SOM (Self-Organizing Maps)
- LSH (Locality-Sensitive Hashing)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import warnings


@dataclass
class ClusterResult:
    """Container for clustering results."""
    algorithm: str
    labels: np.ndarray
    n_clusters: int
    cluster_centers: Optional[np.ndarray]
    noise_points: np.ndarray
    silhouette_score: Optional[float]
    parameters: Dict


class LSHIndex:
    """
    Locality-Sensitive Hashing for approximate nearest neighbor search.

    Uses random hyperplane hashing for efficient similarity search
    in high-dimensional spaces.
    """

    def __init__(self, n_hash_tables: int = 10, n_hash_functions: int = 8,
                 random_state: Optional[int] = None):
        self.n_hash_tables = n_hash_tables
        self.n_hash_functions = n_hash_functions
        self.random_state = random_state
        self.hash_tables: List[Dict] = []
        self.hyperplanes: List[np.ndarray] = []
        self.data: Optional[np.ndarray] = None

    def _generate_hyperplanes(self, n_features: int) -> None:
        """Generate random hyperplanes for hashing."""
        rng = np.random.RandomState(self.random_state)
        self.hyperplanes = [
            rng.randn(self.n_hash_functions, n_features)
            for _ in range(self.n_hash_tables)
        ]

    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> str:
        """Hash a vector using random hyperplanes."""
        projections = np.dot(self.hyperplanes[table_idx], vector)
        hash_bits = (projections >= 0).astype(int)
        return ''.join(map(str, hash_bits))

    def fit(self, data: np.ndarray) -> 'LSHIndex':
        """Build LSH index from data."""
        self.data = data
        n_samples, n_features = data.shape
        self._generate_hyperplanes(n_features)

        self.hash_tables = [{} for _ in range(self.n_hash_tables)]

        for idx, vector in enumerate(data):
            for table_idx in range(self.n_hash_tables):
                hash_key = self._hash_vector(vector, table_idx)
                if hash_key not in self.hash_tables[table_idx]:
                    self.hash_tables[table_idx][hash_key] = []
                self.hash_tables[table_idx][hash_key].append(idx)

        return self

    def query(self, vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Find k approximate nearest neighbors."""
        candidates = set()

        for table_idx in range(self.n_hash_tables):
            hash_key = self._hash_vector(vector, table_idx)
            if hash_key in self.hash_tables[table_idx]:
                candidates.update(self.hash_tables[table_idx][hash_key])

        if len(candidates) == 0:
            # Fallback to brute force if no candidates found
            distances = np.linalg.norm(self.data - vector, axis=1)
            indices = np.argsort(distances)[:k]
            return distances[indices], indices

        candidates = list(candidates)
        candidate_vectors = self.data[candidates]
        distances = np.linalg.norm(candidate_vectors - vector, axis=1)

        k_actual = min(k, len(candidates))
        top_k_idx = np.argsort(distances)[:k_actual]

        return distances[top_k_idx], np.array(candidates)[top_k_idx]

    def get_clusters(self, similarity_threshold: float = 0.5) -> np.ndarray:
        """
        Get cluster assignments based on LSH buckets.
        Points in the same bucket across multiple tables are in the same cluster.
        """
        n_samples = len(self.data)
        labels = np.full(n_samples, -1)
        current_cluster = 0

        # Use connected components approach
        adjacency = {i: set() for i in range(n_samples)}

        for table in self.hash_tables:
            for bucket_indices in table.values():
                for i in bucket_indices:
                    for j in bucket_indices:
                        if i != j:
                            adjacency[i].add(j)

        visited = set()
        for start_idx in range(n_samples):
            if start_idx in visited:
                continue

            # BFS to find connected component
            queue = [start_idx]
            component = []

            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)

                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            for idx in component:
                labels[idx] = current_cluster
            current_cluster += 1

        return labels


class SOMClustering:
    """
    Self-Organizing Map (Kohonen Network) for clustering.

    A neural network-based approach for dimensionality reduction
    and clustering that preserves topological properties.
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10),
                 learning_rate: float = 0.5, sigma: float = 1.0,
                 n_iterations: int = 1000, random_state: Optional[int] = None):
        self.grid_size = grid_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights: Optional[np.ndarray] = None
        self._som = None

    def fit(self, data: np.ndarray) -> 'SOMClustering':
        """Train the SOM on input data."""
        try:
            from minisom import MiniSom

            n_features = data.shape[1]
            self._som = MiniSom(
                self.grid_size[0], self.grid_size[1], n_features,
                sigma=self.sigma, learning_rate=self.learning_rate,
                random_seed=self.random_state
            )
            self._som.random_weights_init(data)
            self._som.train_random(data, self.n_iterations)
            self.weights = self._som.get_weights()

        except ImportError:
            # Fallback implementation if minisom not available
            self._fit_manual(data)

        return self

    def _fit_manual(self, data: np.ndarray) -> None:
        """Manual SOM implementation as fallback."""
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = data.shape

        # Initialize weights
        self.weights = rng.randn(
            self.grid_size[0], self.grid_size[1], n_features
        )

        # Create grid coordinates
        grid_coords = np.array([
            [i, j]
            for i in range(self.grid_size[0])
            for j in range(self.grid_size[1])
        ])

        for iteration in range(self.n_iterations):
            # Decay learning rate and sigma
            lr = self.learning_rate * np.exp(-iteration / self.n_iterations)
            sig = self.sigma * np.exp(-iteration / self.n_iterations)

            # Random sample
            idx = rng.randint(n_samples)
            sample = data[idx]

            # Find Best Matching Unit (BMU)
            bmu_idx = self._find_bmu(sample)

            # Update weights
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    dist_to_bmu = np.sqrt((i - bmu_idx[0])**2 + (j - bmu_idx[1])**2)
                    influence = np.exp(-dist_to_bmu**2 / (2 * sig**2))
                    self.weights[i, j] += lr * influence * (sample - self.weights[i, j])

    def _find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """Find the Best Matching Unit for a sample."""
        min_dist = float('inf')
        bmu = (0, 0)

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                dist = np.linalg.norm(sample - self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu = (i, j)

        return bmu

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Assign cluster labels based on BMU."""
        labels = np.zeros(len(data), dtype=int)

        for idx, sample in enumerate(data):
            if self._som is not None:
                bmu = self._som.winner(sample)
            else:
                bmu = self._find_bmu(sample)
            labels[idx] = bmu[0] * self.grid_size[1] + bmu[1]

        return labels

    def get_cluster_centers(self) -> np.ndarray:
        """Get the weight vectors as cluster centers."""
        return self.weights.reshape(-1, self.weights.shape[-1])


class ClusteringEngine:
    """
    Multi-algorithm clustering engine for predictive maintenance.

    Combines results from K-means, DBSCAN, OPTICS, SOM, and LSH
    to provide robust clustering for anomaly detection.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results: Dict[str, ClusterResult] = {}
        self.data_scaled: Optional[np.ndarray] = None
        self.data_original: Optional[np.ndarray] = None
        self.kmeans_model: Optional[KMeans] = None

    def fit(self, data: np.ndarray,
            algorithms: Optional[List[str]] = None,
            params: Optional[Dict] = None) -> 'ClusteringEngine':
        """
        Fit clustering algorithms to the data.

        Args:
            data: Input data array of shape (n_samples, n_features)
            algorithms: List of algorithms to use. Default is all.
            params: Custom parameters for each algorithm

        Returns:
            Self
        """
        if algorithms is None:
            algorithms = ['kmeans', 'dbscan', 'optics', 'som', 'lsh']

        if params is None:
            params = {}

        self.data_original = data
        self.data_scaled = self.scaler.fit_transform(data)

        for alg in algorithms:
            alg_params = params.get(alg, {})

            if alg == 'kmeans':
                self._fit_kmeans(alg_params)
            elif alg == 'dbscan':
                self._fit_dbscan(alg_params)
            elif alg == 'optics':
                self._fit_optics(alg_params)
            elif alg == 'som':
                self._fit_som(alg_params)
            elif alg == 'lsh':
                self._fit_lsh(alg_params)

        return self

    def _fit_kmeans(self, params: Dict) -> None:
        """Fit K-means clustering."""
        n_clusters = params.get('n_clusters', self._estimate_n_clusters())

        model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        labels = model.fit_predict(self.data_scaled)
        self.kmeans_model = model

        self.results['kmeans'] = ClusterResult(
            algorithm='kmeans',
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=self.scaler.inverse_transform(model.cluster_centers_),
            noise_points=np.array([]),
            silhouette_score=self._compute_silhouette(labels),
            parameters={'n_clusters': n_clusters}
        )

    def _fit_dbscan(self, params: Dict) -> None:
        """Fit DBSCAN clustering."""
        eps = params.get('eps', self._estimate_eps())
        min_samples = params.get('min_samples', max(5, 2 * self.data_scaled.shape[1]))

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(self.data_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_mask = labels == -1

        # Calculate cluster centers (excluding noise)
        centers = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            center = self.data_original[cluster_mask].mean(axis=0)
            centers.append(center)

        self.results['dbscan'] = ClusterResult(
            algorithm='dbscan',
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=np.array(centers) if centers else None,
            noise_points=np.where(noise_mask)[0],
            silhouette_score=self._compute_silhouette(labels) if n_clusters > 1 else None,
            parameters={'eps': eps, 'min_samples': min_samples}
        )

    def _fit_optics(self, params: Dict) -> None:
        """Fit OPTICS clustering."""
        min_samples = params.get('min_samples', max(5, 2 * self.data_scaled.shape[1]))
        xi = params.get('xi', 0.05)

        model = OPTICS(min_samples=min_samples, xi=xi, cluster_method='xi')
        labels = model.fit_predict(self.data_scaled)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_mask = labels == -1

        # Calculate cluster centers
        centers = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() > 0:
                center = self.data_original[cluster_mask].mean(axis=0)
                centers.append(center)

        self.results['optics'] = ClusterResult(
            algorithm='optics',
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=np.array(centers) if centers else None,
            noise_points=np.where(noise_mask)[0],
            silhouette_score=self._compute_silhouette(labels) if n_clusters > 1 else None,
            parameters={'min_samples': min_samples, 'xi': xi}
        )

    def _fit_som(self, params: Dict) -> None:
        """Fit SOM clustering."""
        grid_size = params.get('grid_size', self._estimate_som_grid())
        n_iterations = params.get('n_iterations', 1000)

        som = SOMClustering(
            grid_size=grid_size,
            n_iterations=n_iterations,
            random_state=self.random_state
        )
        som.fit(self.data_scaled)
        labels = som.predict(self.data_scaled)

        # Merge small clusters
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_cluster_size = max(2, len(self.data_scaled) // 50)

        label_mapping = {}
        new_label = 0
        for label, count in zip(unique_labels, counts):
            if count >= min_cluster_size:
                label_mapping[label] = new_label
                new_label += 1
            else:
                label_mapping[label] = -1  # Mark as noise

        labels = np.array([label_mapping[l] for l in labels])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        self.results['som'] = ClusterResult(
            algorithm='som',
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=self.scaler.inverse_transform(som.get_cluster_centers()),
            noise_points=np.where(labels == -1)[0],
            silhouette_score=self._compute_silhouette(labels) if n_clusters > 1 else None,
            parameters={'grid_size': grid_size, 'n_iterations': n_iterations}
        )

    def _fit_lsh(self, params: Dict) -> None:
        """Fit LSH-based clustering."""
        n_hash_tables = params.get('n_hash_tables', 10)
        n_hash_functions = params.get('n_hash_functions', 8)

        lsh = LSHIndex(
            n_hash_tables=n_hash_tables,
            n_hash_functions=n_hash_functions,
            random_state=self.random_state
        )
        lsh.fit(self.data_scaled)
        labels = lsh.get_clusters()

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        # Calculate cluster centers
        centers = []
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            if cluster_mask.sum() > 0:
                center = self.data_original[cluster_mask].mean(axis=0)
                centers.append(center)

        self.results['lsh'] = ClusterResult(
            algorithm='lsh',
            labels=labels,
            n_clusters=n_clusters,
            cluster_centers=np.array(centers) if centers else None,
            noise_points=np.where(labels == -1)[0],
            silhouette_score=self._compute_silhouette(labels) if n_clusters > 1 else None,
            parameters={'n_hash_tables': n_hash_tables, 'n_hash_functions': n_hash_functions}
        )

    def _estimate_n_clusters(self) -> int:
        """Estimate optimal number of clusters using elbow method."""
        max_clusters = min(10, len(self.data_scaled) // 5)
        max_clusters = max(2, max_clusters)

        inertias = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            kmeans.fit(self.data_scaled)
            inertias.append(kmeans.inertia_)

        # Simple elbow detection
        if len(inertias) < 2:
            return 3

        diffs = np.diff(inertias)
        elbow_idx = np.argmax(np.abs(np.diff(diffs))) + 2
        return min(elbow_idx + 2, max_clusters)

    def _estimate_eps(self) -> float:
        """Estimate DBSCAN eps using k-distance graph."""
        k = min(5, len(self.data_scaled) - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.data_scaled)
        distances, _ = nn.kneighbors(self.data_scaled)

        k_distances = np.sort(distances[:, -1])
        # Find knee point
        n = len(k_distances)
        coords = np.column_stack([np.arange(n), k_distances])
        line_vec = coords[-1] - coords[0]
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        vec_from_first = coords - coords[0]
        proj = np.dot(vec_from_first, line_vec_norm)
        proj_points = np.outer(proj, line_vec_norm) + coords[0]
        distances_to_line = np.linalg.norm(coords - proj_points, axis=1)

        knee_idx = np.argmax(distances_to_line)
        return k_distances[knee_idx]

    def _estimate_som_grid(self) -> Tuple[int, int]:
        """Estimate SOM grid size based on data size."""
        n_samples = len(self.data_scaled)
        grid_dim = int(np.sqrt(np.sqrt(n_samples)))
        grid_dim = max(3, min(grid_dim, 15))
        return (grid_dim, grid_dim)

    def _compute_silhouette(self, labels: np.ndarray) -> Optional[float]:
        """Compute silhouette score for clustering."""
        from sklearn.metrics import silhouette_score

        unique_labels = set(labels)
        unique_labels.discard(-1)

        if len(unique_labels) < 2:
            return None

        # Only use non-noise points
        mask = labels != -1
        if mask.sum() < 2:
            return None

        try:
            return silhouette_score(self.data_scaled[mask], labels[mask])
        except Exception:
            return None

    def get_ensemble_labels(self, method: str = 'voting') -> np.ndarray:
        """
        Get ensemble cluster labels from all algorithms.

        Args:
            method: 'voting' for majority voting, 'consensus' for consensus clustering

        Returns:
            Ensemble cluster labels
        """
        from scipy.cluster.hierarchy import linkage, fcluster

        if len(self.results) == 0:
            raise ValueError("No clustering results available. Run fit() first.")

        n_samples = len(self.data_scaled)

        # Build co-association matrix
        coassoc = np.zeros((n_samples, n_samples))

        for result in self.results.values():
            labels = result.labels
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] >= 0 and labels[j] >= 0 and labels[i] == labels[j]:
                        coassoc[i, j] += 1
                        coassoc[j, i] += 1

        coassoc /= len(self.results)

        # Convert to distance matrix
        distance_matrix = 1 - coassoc

        # Hierarchical clustering on co-association matrix
        condensed_dist = distance_matrix[np.triu_indices(n_samples, k=1)]
        Z = linkage(condensed_dist, method='average')

        # Determine number of clusters
        avg_clusters = int(np.mean([r.n_clusters for r in self.results.values() if r.n_clusters > 0]))
        avg_clusters = max(2, avg_clusters)

        return fcluster(Z, t=avg_clusters, criterion='maxclust') - 1

    def get_result(self, algorithm: str) -> ClusterResult:
        """Get clustering result for a specific algorithm."""
        if algorithm not in self.results:
            raise KeyError(f"Algorithm '{algorithm}' not found. Available: {list(self.results.keys())}")
        return self.results[algorithm]

    def get_all_results(self) -> Dict[str, ClusterResult]:
        """Get all clustering results."""
        return self.results
