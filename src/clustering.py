"""
Clustering module for verb semantic class induction.

Implements multiple clustering methods:
- Hierarchical (agglomerative) clustering
- Graph-based clustering (Louvain, Chinese Whispers)
- K-Means, HDBSCAN, Spectral clustering

Supports comparative evaluation across methods and feature types.
"""

from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
import json
import warnings

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Optional imports for advanced clustering
try:
    from sklearn.cluster import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    import networkx as nx
    from community import community_louvain
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False


@dataclass
class ClusteringResult:
    """Result of a single clustering run."""
    method: str
    feature_type: str
    n_clusters: int
    labels: list[int]
    
    # Intrinsic metrics
    silhouette: Optional[float] = None
    calinski_harabasz: Optional[float] = None
    davies_bouldin: Optional[float] = None
    modularity: Optional[float] = None  # For graph-based methods
    
    # Metadata
    params: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        def to_python(x):
            """Convert numpy types to Python types."""
            if x is None:
                return None
            if isinstance(x, (np.floating, np.integer)):
                return float(x)
            return x
        
        return {
            "method": self.method,
            "feature_type": self.feature_type,
            "n_clusters": int(self.n_clusters),
            "silhouette": to_python(self.silhouette),
            "calinski_harabasz": to_python(self.calinski_harabasz),
            "davies_bouldin": to_python(self.davies_bouldin),
            "modularity": to_python(self.modularity),
            "params": self.params,
        }


def compute_intrinsic_metrics(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """
    Compute intrinsic clustering metrics.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        labels: Cluster labels
        
    Returns:
        Dict of metric names to values
    """
    # Need at least 2 clusters for meaningful metrics
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
        }
    
    # Filter out noise points (label -1) for metrics
    mask = labels != -1
    if mask.sum() < 2:
        return {
            "silhouette": None,
            "calinski_harabasz": None,
            "davies_bouldin": None,
        }
    
    features_clean = features[mask]
    labels_clean = labels[mask]
    
    metrics = {}
    
    try:
        metrics["silhouette"] = float(silhouette_score(features_clean, labels_clean))
    except Exception:
        metrics["silhouette"] = None
    
    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(features_clean, labels_clean))
    except Exception:
        metrics["calinski_harabasz"] = None
    
    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(features_clean, labels_clean))
    except Exception:
        metrics["davies_bouldin"] = None
    
    return metrics


# =============================================================================
# Hierarchical Clustering
# =============================================================================

def hierarchical_clustering(
    features: np.ndarray,
    n_clusters: int,
    method: str = "ward",
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Perform agglomerative hierarchical clustering.
    
    Args:
        features: Feature matrix
        n_clusters: Number of clusters to form
        method: Linkage method ('ward', 'average', 'complete', 'single')
        metric: Distance metric
        
    Returns:
        Cluster labels
    """
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric if method != "ward" else "euclidean",
        linkage=method,
    )
    return clustering.fit_predict(features)


def compute_linkage_and_dendrogram(
    features: np.ndarray,
    method: str = "ward",
    output_path: Optional[Path] = None,
    title: str = "Hierarchical Clustering Dendrogram",
    max_display: int = 50,
    labels: Optional[list[str]] = None,
) -> np.ndarray:
    """
    Compute linkage matrix and optionally save dendrogram.
    
    Args:
        features: Feature matrix
        method: Linkage method
        output_path: Path to save dendrogram image
        title: Plot title
        max_display: Max leaf nodes to display
        labels: Optional list of labels for leaf nodes
        
    Returns:
        Linkage matrix
    """
    # Compute linkage
    if method == "ward":
        Z = linkage(features, method="ward")
    else:
        Z = linkage(features, method=method, metric="cosine")
    
    if output_path:
        n_samples = features.shape[0]
        
        if n_samples <= max_display and labels:
            # Show full dendrogram with labels
            fig, ax = plt.subplots(figsize=(max(16, n_samples * 0.15), 10))
            
            dendrogram(
                Z,
                labels=labels,
                leaf_rotation=90,
                leaf_font_size=7,
                ax=ax,
            )
        else:
            # Truncated view for large datasets - show representative labels
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Get cluster assignments at truncation level
            cluster_labels = fcluster(Z, max_display, criterion="maxclust")
            
            # Build representative labels for each cluster
            if labels:
                cluster_reps = {}
                for i, (label, cluster) in enumerate(zip(labels, cluster_labels)):
                    if cluster not in cluster_reps:
                        cluster_reps[cluster] = []
                    cluster_reps[cluster].append(label)
                
                # Create labels showing cluster representatives
                def label_func(id):
                    if id < n_samples:
                        return labels[id]
                    return ""
                
                dendrogram(
                    Z,
                    truncate_mode="lastp",
                    p=max_display,
                    leaf_rotation=90,
                    leaf_font_size=8,
                    ax=ax,
                    show_leaf_counts=True,
                )
                
                # Add annotation with top verbs per cluster
                annotation_text = "Top verbs per cluster:\n"
                for c in sorted(cluster_reps.keys())[:10]:
                    reps = cluster_reps[c][:5]  # Top 5
                    annotation_text += f"  C{c}: {', '.join(reps)}\n"
                
                ax.text(
                    0.02, 0.98, annotation_text,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                )
            else:
                dendrogram(
                    Z,
                    truncate_mode="lastp",
                    p=max_display,
                    leaf_rotation=90,
                    leaf_font_size=8,
                    ax=ax,
                )
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Verb Lemmas / Cluster", fontsize=10)
        ax.set_ylabel("Distance", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return Z


def hierarchical_at_k_values(
    features: np.ndarray,
    k_values: list[int],
    method: str = "ward",
) -> dict[int, np.ndarray]:
    """
    Compute hierarchical clustering at multiple k values.
    
    Args:
        features: Feature matrix
        k_values: List of cluster counts
        method: Linkage method
        
    Returns:
        Dict of k -> labels
    """
    # Compute linkage once
    if method == "ward":
        Z = linkage(features, method="ward")
    else:
        Z = linkage(features, method=method, metric="cosine")
    
    results = {}
    for k in k_values:
        labels = fcluster(Z, k, criterion="maxclust") - 1  # 0-indexed
        results[k] = labels
    
    return results


# =============================================================================
# Graph-Based Clustering
# =============================================================================

def build_knn_graph(
    features: np.ndarray,
    k: int = 15,
    metric: str = "cosine",
) -> "nx.Graph":
    """
    Build k-nearest neighbor graph from features.
    
    Args:
        features: Feature matrix
        k: Number of neighbors
        metric: Distance metric
        
    Returns:
        NetworkX graph with similarity weights
    """
    if not HAS_GRAPH:
        raise ImportError("networkx and python-louvain required for graph clustering")
    
    from sklearn.neighbors import NearestNeighbors
    
    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    
    # Build graph
    G = nx.Graph()
    n_samples = features.shape[0]
    G.add_nodes_from(range(n_samples))
    
    # Add edges with similarity weights
    for i in range(n_samples):
        for j, dist in zip(indices[i, 1:], distances[i, 1:]):  # Skip self
            # Convert distance to similarity
            if metric == "cosine":
                similarity = 1 - dist
            else:
                similarity = 1 / (1 + dist)
            
            if similarity > 0:
                # Add edge (or update if exists)
                if G.has_edge(i, j):
                    G[i][j]["weight"] = max(G[i][j]["weight"], similarity)
                else:
                    G.add_edge(i, j, weight=similarity)
    
    return G


def louvain_clustering(
    features: np.ndarray,
    k_neighbors: int = 15,
    resolution: float = 1.0,
) -> tuple[np.ndarray, float]:
    """
    Perform Louvain community detection on k-NN graph.
    
    Args:
        features: Feature matrix
        k_neighbors: Number of neighbors for graph construction
        resolution: Resolution parameter (higher = more clusters)
        
    Returns:
        Tuple of (labels, modularity)
    """
    if not HAS_GRAPH:
        raise ImportError("networkx and python-louvain required")
    
    G = build_knn_graph(features, k=k_neighbors)
    
    # Run Louvain
    partition = community_louvain.best_partition(G, resolution=resolution)
    
    # Convert to labels array
    labels = np.array([partition[i] for i in range(len(partition))])
    
    # Compute modularity
    modularity = community_louvain.modularity(partition, G)
    
    return labels, modularity


def chinese_whispers(
    features: np.ndarray,
    k_neighbors: int = 15,
    iterations: int = 20,
    seed: int = 42,
) -> np.ndarray:
    """
    Chinese Whispers clustering algorithm.
    
    A simple, parameter-free graph clustering algorithm.
    
    Args:
        features: Feature matrix
        k_neighbors: Number of neighbors for graph construction
        iterations: Number of iterations
        seed: Random seed
        
    Returns:
        Cluster labels
    """
    if not HAS_GRAPH:
        raise ImportError("networkx required")
    
    np.random.seed(seed)
    
    G = build_knn_graph(features, k=k_neighbors)
    n_nodes = G.number_of_nodes()
    
    # Initialize: each node in its own cluster
    labels = {i: i for i in range(n_nodes)}
    
    # Iterate
    for _ in range(iterations):
        # Process nodes in random order
        nodes = list(G.nodes())
        np.random.shuffle(nodes)
        
        for node in nodes:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                continue
            
            # Count weighted votes for each label
            label_weights = {}
            for neighbor in neighbors:
                weight = G[node][neighbor].get("weight", 1.0)
                label = labels[neighbor]
                label_weights[label] = label_weights.get(label, 0) + weight
            
            # Assign most popular label
            if label_weights:
                labels[node] = max(label_weights, key=label_weights.get)
    
    # Convert to contiguous labels
    unique_labels = sorted(set(labels.values()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    
    return np.array([label_map[labels[i]] for i in range(n_nodes)])


# =============================================================================
# Other Clustering Methods
# =============================================================================

def kmeans_clustering(
    features: np.ndarray,
    n_clusters: int,
    n_init: int = 10,
    seed: int = 42,
) -> np.ndarray:
    """K-Means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    return kmeans.fit_predict(features)


def hdbscan_clustering(
    features: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = None,
) -> np.ndarray:
    """HDBSCAN clustering (density-based)."""
    if not HAS_HDBSCAN:
        raise ImportError("HDBSCAN not available in this sklearn version")
    
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    return clusterer.fit_predict(features)


def spectral_clustering(
    features: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    """Spectral clustering."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity="nearest_neighbors",
            n_neighbors=10,
            random_state=seed,
        )
        return clustering.fit_predict(features)


# =============================================================================
# Comprehensive Clustering Pipeline
# =============================================================================

@dataclass
class ClusteringConfig:
    """Configuration for clustering experiments."""
    k_values: list[int] = field(default_factory=lambda: [10, 20, 30, 50])
    hierarchical_methods: list[str] = field(default_factory=lambda: ["ward", "average"])
    louvain_resolutions: list[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    knn_k: int = 15
    hdbscan_min_sizes: list[int] = field(default_factory=lambda: [5, 10])
    seed: int = 42


def run_clustering_experiments(
    features: np.ndarray,
    feature_type: str,
    config: Optional[ClusteringConfig] = None,
) -> list[ClusteringResult]:
    """
    Run comprehensive clustering experiments on features.
    
    Args:
        features: Feature matrix
        feature_type: Name of feature type (e.g., "syntactic", "embedding", "joint")
        config: Clustering configuration
        
    Returns:
        List of ClusteringResult objects
    """
    config = config or ClusteringConfig()
    results = []
    
    print(f"\nClustering with {feature_type} features ({features.shape})")
    print("-" * 50)
    
    # 1. Hierarchical clustering at multiple k values
    for method in config.hierarchical_methods:
        print(f"  Hierarchical ({method})...")
        k_results = hierarchical_at_k_values(features, config.k_values, method=method)
        
        for k, labels in k_results.items():
            metrics = compute_intrinsic_metrics(features, labels)
            result = ClusteringResult(
                method=f"hierarchical_{method}",
                feature_type=feature_type,
                n_clusters=k,
                labels=labels.tolist(),
                silhouette=metrics["silhouette"],
                calinski_harabasz=metrics["calinski_harabasz"],
                davies_bouldin=metrics["davies_bouldin"],
                params={"linkage": method, "k": k},
            )
            results.append(result)
    
    # 2. K-Means at multiple k values
    print("  K-Means...")
    for k in config.k_values:
        labels = kmeans_clustering(features, k, seed=config.seed)
        metrics = compute_intrinsic_metrics(features, labels)
        result = ClusteringResult(
            method="kmeans",
            feature_type=feature_type,
            n_clusters=k,
            labels=labels.tolist(),
            silhouette=metrics["silhouette"],
            calinski_harabasz=metrics["calinski_harabasz"],
            davies_bouldin=metrics["davies_bouldin"],
            params={"k": k},
        )
        results.append(result)
    
    # 3. Graph-based clustering
    if HAS_GRAPH:
        # Louvain at multiple resolutions
        print("  Louvain...")
        for resolution in config.louvain_resolutions:
            try:
                labels, modularity = louvain_clustering(
                    features, 
                    k_neighbors=config.knn_k,
                    resolution=resolution,
                )
                metrics = compute_intrinsic_metrics(features, labels)
                n_clusters = len(set(labels))
                result = ClusteringResult(
                    method="louvain",
                    feature_type=feature_type,
                    n_clusters=n_clusters,
                    labels=labels.tolist(),
                    silhouette=metrics["silhouette"],
                    calinski_harabasz=metrics["calinski_harabasz"],
                    davies_bouldin=metrics["davies_bouldin"],
                    modularity=modularity,
                    params={"resolution": resolution, "k_neighbors": config.knn_k},
                )
                results.append(result)
            except Exception as e:
                print(f"    Louvain (res={resolution}) failed: {e}")
        
        # Chinese Whispers
        print("  Chinese Whispers...")
        try:
            labels = chinese_whispers(features, k_neighbors=config.knn_k, seed=config.seed)
            metrics = compute_intrinsic_metrics(features, labels)
            n_clusters = len(set(labels))
            result = ClusteringResult(
                method="chinese_whispers",
                feature_type=feature_type,
                n_clusters=n_clusters,
                labels=labels.tolist(),
                silhouette=metrics["silhouette"],
                calinski_harabasz=metrics["calinski_harabasz"],
                davies_bouldin=metrics["davies_bouldin"],
                params={"k_neighbors": config.knn_k},
            )
            results.append(result)
        except Exception as e:
            print(f"    Chinese Whispers failed: {e}")
    
    # 4. HDBSCAN
    if HAS_HDBSCAN:
        print("  HDBSCAN...")
        for min_size in config.hdbscan_min_sizes:
            try:
                labels = hdbscan_clustering(features, min_cluster_size=min_size)
                # Count non-noise clusters
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    metrics = compute_intrinsic_metrics(features, labels)
                    result = ClusteringResult(
                        method="hdbscan",
                        feature_type=feature_type,
                        n_clusters=n_clusters,
                        labels=labels.tolist(),
                        silhouette=metrics["silhouette"],
                        calinski_harabasz=metrics["calinski_harabasz"],
                        davies_bouldin=metrics["davies_bouldin"],
                        params={"min_cluster_size": min_size},
                    )
                    results.append(result)
            except Exception as e:
                print(f"    HDBSCAN (min_size={min_size}) failed: {e}")
    
    # 5. Spectral clustering
    print("  Spectral...")
    for k in config.k_values[:2]:  # Only a few k values (slow)
        try:
            labels = spectral_clustering(features, k, seed=config.seed)
            metrics = compute_intrinsic_metrics(features, labels)
            result = ClusteringResult(
                method="spectral",
                feature_type=feature_type,
                n_clusters=k,
                labels=labels.tolist(),
                silhouette=metrics["silhouette"],
                calinski_harabasz=metrics["calinski_harabasz"],
                davies_bouldin=metrics["davies_bouldin"],
                params={"k": k},
            )
            results.append(result)
        except Exception as e:
            print(f"    Spectral (k={k}) failed: {e}")
    
    return results


def find_best_clustering(
    results: list[ClusteringResult],
    metric: str = "silhouette",
) -> ClusteringResult:
    """Find best clustering result by given metric."""
    valid_results = [r for r in results if getattr(r, metric) is not None]
    if not valid_results:
        return results[0]
    
    if metric == "davies_bouldin":
        # Lower is better
        return min(valid_results, key=lambda r: getattr(r, metric))
    else:
        # Higher is better
        return max(valid_results, key=lambda r: getattr(r, metric))


def generate_comparison_table(
    results: list[ClusteringResult],
) -> str:
    """Generate markdown comparison table."""
    lines = [
        "| Method | Feature | k | Silhouette | CH Index | DB Index | Modularity |",
        "|--------|---------|---|------------|----------|----------|------------|",
    ]
    
    # Sort by silhouette (descending)
    sorted_results = sorted(
        results, 
        key=lambda r: r.silhouette if r.silhouette else -1,
        reverse=True,
    )
    
    for r in sorted_results[:30]:  # Top 30
        sil = f"{r.silhouette:.3f}" if r.silhouette else "N/A"
        ch = f"{r.calinski_harabasz:.1f}" if r.calinski_harabasz else "N/A"
        db = f"{r.davies_bouldin:.3f}" if r.davies_bouldin else "N/A"
        mod = f"{r.modularity:.3f}" if r.modularity else "-"
        lines.append(f"| {r.method} | {r.feature_type} | {r.n_clusters} | {sil} | {ch} | {db} | {mod} |")
    
    return "\n".join(lines)


# =============================================================================
# Main Pipeline
# =============================================================================

def run_english_clustering(
    output_dir: Path,
    syntactic_path: Path,
    embedding_path: Path,
    joint_path: Path,
    lemma_index_path: Path,
) -> dict:
    """
    Run full clustering pipeline for English.
    
    Args:
        output_dir: Output directory
        syntactic_path: Path to syntactic features .npz
        embedding_path: Path to embedding features .npz
        joint_path: Path to joint features .npz
        lemma_index_path: Path to lemma index .json
        
    Returns:
        Summary dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Phase 4: Clustering and Intrinsic Evaluation (English)")
    print("=" * 60)
    
    # Load features
    print("\nLoading features...")
    syn_features = np.load(syntactic_path)["features"]
    emb_features = np.load(embedding_path)["embeddings"]
    joint_features = np.load(joint_path)["features"]
    
    with open(lemma_index_path) as f:
        lemma_index = json.load(f)
    lemmas = lemma_index["index_to_lemma"]
    
    print(f"  Syntactic: {syn_features.shape}")
    print(f"  Embedding: {emb_features.shape}")
    print(f"  Joint: {joint_features.shape}")
    print(f"  Lemmas: {len(lemmas)}")
    
    # Configuration
    config = ClusteringConfig(
        k_values=[10, 20, 30, 50],
        hierarchical_methods=["ward", "average"],
        louvain_resolutions=[0.5, 1.0, 1.5, 2.0],
        hdbscan_min_sizes=[5, 10],
    )
    
    all_results = []
    
    # Run clustering on each feature type
    for name, features in [
        ("syntactic", syn_features),
        ("embedding", emb_features),
        ("joint", joint_features),
    ]:
        results = run_clustering_experiments(features, name, config)
        all_results.extend(results)
    
    # Generate dendrograms
    print("\nGenerating dendrograms...")
    for name, features in [
        ("syntactic", syn_features),
        ("embedding", emb_features),
        ("joint", joint_features),
    ]:
        # Full dendrogram with cluster annotations
        compute_linkage_and_dendrogram(
            features,
            method="ward",
            output_path=output_dir / f"eng_dendrogram_{name}.png",
            title=f"English Verb Clustering ({name.capitalize()} Features) - All {len(lemmas)} Verbs",
            labels=lemmas,
        )
        
        # Also generate a focused dendrogram with top 100 verbs (readable labels)
        top_n = min(100, len(lemmas))
        compute_linkage_and_dendrogram(
            features[:top_n],
            method="ward",
            output_path=output_dir / f"eng_dendrogram_{name}_top100.png",
            title=f"English Verb Clustering ({name.capitalize()}) - Top {top_n} Verbs by Frequency",
            labels=lemmas[:top_n],
            max_display=top_n,
        )
    
    # Find best results
    print("\nBest results by feature type:")
    print("-" * 50)
    
    best_per_feature = {}
    for feature_type in ["syntactic", "embedding", "joint"]:
        type_results = [r for r in all_results if r.feature_type == feature_type]
        best = find_best_clustering(type_results, "silhouette")
        best_per_feature[feature_type] = best
        print(f"  {feature_type}: {best.method} (k={best.n_clusters}, sil={best.silhouette:.3f})")
    
    # Overall best
    overall_best = find_best_clustering(all_results, "silhouette")
    print(f"\n  OVERALL BEST: {overall_best.feature_type}/{overall_best.method}")
    print(f"    k={overall_best.n_clusters}, silhouette={overall_best.silhouette:.3f}")
    
    # Save results
    print("\nSaving results...")
    
    # All results as JSON
    results_data = {
        "all_results": [r.to_dict() for r in all_results],
        "best_per_feature": {k: v.to_dict() for k, v in best_per_feature.items()},
        "overall_best": overall_best.to_dict(),
    }
    
    with open(output_dir / "eng_clustering_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    
    # Comparison table as markdown
    table = generate_comparison_table(all_results)
    with open(output_dir / "eng_intrinsic_scores.md", "w") as f:
        f.write("# English Verb Clustering: Intrinsic Evaluation\n\n")
        f.write("Sorted by Silhouette score (higher is better).\n\n")
        f.write(table)
    
    # Save best cluster assignments
    for feature_type, result in best_per_feature.items():
        assignments = {
            lemmas[i]: result.labels[i] 
            for i in range(len(lemmas))
        }
        with open(output_dir / f"eng_cluster_assignments_{feature_type}.json", "w") as f:
            json.dump({
                "method": result.method,
                "n_clusters": result.n_clusters,
                "assignments": assignments,
                "silhouette": result.silhouette,
            }, f, indent=2)
    
    # Generate metrics plot
    plot_metrics_comparison(all_results, output_dir / "eng_clustering_comparison.png")
    
    print(f"\nResults saved to {output_dir}")
    
    return results_data


def plot_metrics_comparison(
    results: list[ClusteringResult],
    output_path: Path,
):
    """Plot comparison of clustering metrics across methods and features."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    feature_types = ["syntactic", "embedding", "joint"]
    colors = {"syntactic": "#e74c3c", "embedding": "#3498db", "joint": "#2ecc71"}
    
    for ax, metric in zip(axes, ["silhouette", "calinski_harabasz", "davies_bouldin"]):
        for ft in feature_types:
            ft_results = [r for r in results if r.feature_type == ft and getattr(r, metric)]
            if not ft_results:
                continue
            
            # Group by method, plot best per method
            methods = {}
            for r in ft_results:
                base_method = r.method.split("_")[0]
                if base_method not in methods:
                    methods[base_method] = r
                elif metric == "davies_bouldin":
                    if getattr(r, metric) < getattr(methods[base_method], metric):
                        methods[base_method] = r
                else:
                    if getattr(r, metric) > getattr(methods[base_method], metric):
                        methods[base_method] = r
            
            x = list(methods.keys())
            y = [getattr(methods[m], metric) for m in x]
            
            ax.bar(
                [f"{m}\n({ft[:3]})" for m in x],
                y,
                alpha=0.7,
                color=colors[ft],
                label=ft if ax == axes[0] else None,
            )
        
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis='x', rotation=45)
    
    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# CLI interface
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"
    
    # Check for joint features (indicates Phase 3b completed)
    joint_path = output_dir / "english_joint_features.npz"
    emb_path = output_dir / "english_verb_ctx_embeddings.npz"
    syn_path = output_dir / "english_verb_features.npz"
    lemma_path = output_dir / "english_lemma_index.json"
    
    if not joint_path.exists():
        print("ERROR: Joint features not found. Run Phase 3b first.")
        exit(1)
    
    results = run_english_clustering(
        output_dir=output_dir,
        syntactic_path=syn_path,
        embedding_path=emb_path,
        joint_path=joint_path,
        lemma_index_path=lemma_path,
    )
    
    print("\n" + "=" * 60)
    print("Clustering complete!")
    print("=" * 60)

