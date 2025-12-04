"""
Cluster analysis and prototype identification.

Analyzes clustering results to:
- Identify prototype verbs (closest to centroid)
- Characterize argument patterns per cluster
- Compare syntactic vs semantic clustering
"""

from pathlib import Path
from typing import Optional
from collections import defaultdict
import json

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, AgglomerativeClustering

from verb_extractor import load_verb_instances, VerbInstance
from feature_engineer import load_features
from clustering import (
    compute_intrinsic_metrics,
    louvain_clustering,
    hierarchical_clustering,
    ClusteringResult,
    HAS_GRAPH,
)


def find_cluster_prototypes(
    features: np.ndarray,
    labels: np.ndarray,
    lemmas: list[str],
    n_prototypes: int = 5,
) -> dict[int, list[tuple[str, float]]]:
    """
    Find prototype verbs for each cluster (closest to centroid).
    
    Args:
        features: Feature matrix
        labels: Cluster labels
        lemmas: Verb lemmas
        n_prototypes: Number of prototypes per cluster
        
    Returns:
        Dict of cluster_id -> [(lemma, distance), ...]
    """
    unique_labels = sorted(set(labels))
    prototypes = {}
    
    for label in unique_labels:
        if label == -1:  # Skip noise
            continue
            
        # Get indices for this cluster
        mask = labels == label
        cluster_indices = np.where(mask)[0]
        cluster_features = features[mask]
        
        # Compute centroid
        centroid = cluster_features.mean(axis=0, keepdims=True)
        
        # Compute distances to centroid
        distances = cdist(cluster_features, centroid, metric='euclidean').flatten()
        
        # Get closest verbs
        sorted_idx = np.argsort(distances)
        top_indices = sorted_idx[:n_prototypes]
        
        prototypes[label] = [
            (lemmas[cluster_indices[i]], float(distances[i]))
            for i in top_indices
        ]
    
    return prototypes


def find_cluster_periphery(
    features: np.ndarray,
    labels: np.ndarray,
    lemmas: list[str],
    n_periphery: int = 5,
) -> dict[int, list[tuple[str, float]]]:
    """
    Find peripheral verbs for each cluster (furthest from centroid).
    These are boundary/ambiguous cases.
    """
    unique_labels = sorted(set(labels))
    periphery = {}
    
    for label in unique_labels:
        if label == -1:
            continue
            
        mask = labels == label
        cluster_indices = np.where(mask)[0]
        cluster_features = features[mask]
        
        if len(cluster_indices) <= n_periphery:
            continue
        
        centroid = cluster_features.mean(axis=0, keepdims=True)
        distances = cdist(cluster_features, centroid, metric='euclidean').flatten()
        
        sorted_idx = np.argsort(distances)[::-1]  # Descending
        top_indices = sorted_idx[:n_periphery]
        
        periphery[label] = [
            (lemmas[cluster_indices[i]], float(distances[i]))
            for i in top_indices
        ]
    
    return periphery


def analyze_cluster_composition(
    labels: np.ndarray,
    lemmas: list[str],
    instances_by_lemma: dict[str, list[VerbInstance]],
) -> dict[int, dict]:
    """
    Analyze argument pattern composition of each cluster.
    """
    unique_labels = sorted(set(labels))
    compositions = {}
    
    for label in unique_labels:
        if label == -1:
            continue
        
        mask = labels == label
        cluster_lemmas = [lemmas[i] for i in range(len(lemmas)) if mask[i]]
        
        # Aggregate transitivity patterns
        trans_counts = defaultdict(int)
        arg_counts = defaultdict(int)
        total_instances = 0
        
        for lemma in cluster_lemmas:
            if lemma in instances_by_lemma:
                for inst in instances_by_lemma[lemma]:
                    total_instances += 1
                    trans_counts[inst.transitivity_pattern] += 1
                    
                    if inst.has_subject:
                        arg_counts["has_subject"] += 1
                    if inst.has_object:
                        arg_counts["has_object"] += 1
                    if inst.has_oblique:
                        arg_counts["has_oblique"] += 1
                    if inst.has_clausal_complement:
                        arg_counts["has_clausal"] += 1
        
        compositions[label] = {
            "n_lemmas": len(cluster_lemmas),
            "n_instances": total_instances,
            "lemmas": cluster_lemmas,
            "transitivity": dict(trans_counts),
            "argument_rates": {
                k: v / total_instances if total_instances > 0 else 0
                for k, v in arg_counts.items()
            },
        }
    
    return compositions


def experiment_embedding_weights(
    syn_features: np.ndarray,
    emb_features: np.ndarray,
    lemmas: list[str],
    alphas: list[float] = [0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
    k_values: list[int] = [15, 20, 30],
) -> list[dict]:
    """
    Experiment with different embedding weights in joint features.
    
    Args:
        syn_features: Syntactic features
        emb_features: Embedding features
        lemmas: Verb lemmas
        alphas: Embedding weights to try (1.0 = embedding only)
        k_values: Cluster counts to try
        
    Returns:
        List of experiment results
    """
    results = []
    
    # Normalize features
    syn_norm = syn_features / (np.linalg.norm(syn_features, axis=1, keepdims=True) + 1e-8)
    emb_norm = emb_features / (np.linalg.norm(emb_features, axis=1, keepdims=True) + 1e-8)
    
    for alpha in alphas:
        # Build joint features with this alpha
        if alpha == 1.0:
            joint = emb_norm
            name = "embedding_only"
        elif alpha == 0.0:
            joint = syn_norm
            name = "syntactic_only"
        else:
            joint = np.hstack([
                alpha * emb_norm,
                (1 - alpha) * syn_norm,
            ])
            # Re-normalize
            joint = joint / (np.linalg.norm(joint, axis=1, keepdims=True) + 1e-8)
            name = f"joint_alpha{alpha}"
        
        for k in k_values:
            # Try hierarchical
            labels = hierarchical_clustering(joint, k, method="ward")
            metrics = compute_intrinsic_metrics(joint, labels)
            
            # Find prototypes
            prototypes = find_cluster_prototypes(joint, labels, lemmas, n_prototypes=5)
            
            results.append({
                "name": name,
                "alpha": alpha,
                "k": k,
                "method": "hierarchical_ward",
                "silhouette": metrics["silhouette"],
                "calinski_harabasz": metrics["calinski_harabasz"],
                "davies_bouldin": metrics["davies_bouldin"],
                "prototypes": {
                    c: [v[0] for v in verbs]  # Just lemmas
                    for c, verbs in prototypes.items()
                },
            })
            
            # Try Louvain if available
            if HAS_GRAPH and alpha > 0:
                for resolution in [0.5, 1.0, 2.0]:
                    try:
                        labels, modularity = louvain_clustering(
                            joint, k_neighbors=15, resolution=resolution
                        )
                        n_clusters = len(set(labels))
                        metrics = compute_intrinsic_metrics(joint, labels)
                        prototypes = find_cluster_prototypes(joint, labels, lemmas, n_prototypes=5)
                        
                        results.append({
                            "name": name,
                            "alpha": alpha,
                            "k": n_clusters,
                            "method": f"louvain_res{resolution}",
                            "silhouette": metrics["silhouette"],
                            "modularity": modularity,
                            "prototypes": {
                                c: [v[0] for v in verbs]
                                for c, verbs in prototypes.items()
                            },
                        })
                    except:
                        pass
    
    return results


def print_cluster_analysis(
    features: np.ndarray,
    labels: np.ndarray,
    lemmas: list[str],
    title: str = "Cluster Analysis",
):
    """Print formatted cluster analysis."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    
    # Find prototypes
    prototypes = find_cluster_prototypes(features, labels, lemmas)
    periphery = find_cluster_periphery(features, labels, lemmas)
    
    # Count cluster sizes
    unique, counts = np.unique(labels[labels != -1], return_counts=True)
    
    for label in sorted(prototypes.keys()):
        size = counts[unique == label][0] if label in unique else 0
        print(f"\n--- Cluster {label} ({size} verbs) ---")
        
        print("  Prototypes (closest to centroid):")
        for verb, dist in prototypes[label]:
            print(f"    • {verb}")
        
        if label in periphery:
            print("  Periphery (boundary cases):")
            for verb, dist in periphery[label]:
                print(f"    ○ {verb}")
    
    # Noise points
    noise_mask = labels == -1
    if noise_mask.any():
        noise_verbs = [lemmas[i] for i in range(len(lemmas)) if noise_mask[i]]
        print(f"\n--- Noise/Outliers ({len(noise_verbs)} verbs) ---")
        print(f"  {', '.join(noise_verbs[:20])}{'...' if len(noise_verbs) > 20 else ''}")


def run_enhanced_clustering(
    output_dir: Path,
    syn_path: Path,
    emb_path: Path,
    lemma_index_path: Path,
    instances_path: Path,
):
    """
    Run enhanced clustering analysis with prototype identification
    and embedding weight experiments.
    """
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("Enhanced Cluster Analysis")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    syn_features = np.load(syn_path)["features"]
    emb_features = np.load(emb_path)["embeddings"]
    
    with open(lemma_index_path) as f:
        lemma_index = json.load(f)
    lemmas = lemma_index["index_to_lemma"]
    
    instances = load_verb_instances(instances_path)
    instances_by_lemma = defaultdict(list)
    for inst in instances:
        instances_by_lemma[inst.lemma].append(inst)
    
    print(f"  Syntactic: {syn_features.shape}")
    print(f"  Embedding: {emb_features.shape}")
    print(f"  Lemmas: {len(lemmas)}")
    
    # Experiment with different alpha values
    print("\n" + "=" * 60)
    print("Experimenting with embedding weights...")
    print("=" * 60)
    
    experiments = experiment_embedding_weights(
        syn_features, emb_features, lemmas,
        alphas=[0.0, 0.5, 0.7, 0.9, 0.95, 1.0],
        k_values=[20, 30],
    )
    
    # Sort by silhouette
    experiments.sort(key=lambda x: x.get("silhouette") or -1, reverse=True)
    
    print("\nTop 10 configurations by silhouette:")
    print("-" * 80)
    print(f"{'Config':<25} {'Method':<20} {'k':>4} {'Silhouette':>10} {'Mod':>6}")
    print("-" * 80)
    
    for exp in experiments[:10]:
        sil = f"{exp['silhouette']:.3f}" if exp.get('silhouette') else 'N/A'
        mod = f"{exp.get('modularity', 0):.3f}" if exp.get('modularity') else '-'
        print(f"{exp['name']:<25} {exp['method']:<20} {exp['k']:>4} {sil:>10} {mod:>6}")
    
    # Show best embedding-heavy configuration
    emb_heavy = [e for e in experiments if e['alpha'] >= 0.9]
    if emb_heavy:
        best_emb = max(emb_heavy, key=lambda x: x.get('silhouette') or -1)
        print(f"\n\nBest embedding-heavy (α≥0.9): {best_emb['name']}, {best_emb['method']}")
        print(f"  k={best_emb['k']}, silhouette={best_emb.get('silhouette', 'N/A')}")
        print("\n  Cluster prototypes:")
        for c, verbs in sorted(best_emb['prototypes'].items()):
            print(f"    Cluster {c}: {', '.join(verbs)}")
    
    # Run detailed analysis on high-alpha joint features
    print("\n" + "=" * 60)
    print("Detailed Analysis: α=0.95 (embedding-dominant)")
    print("=" * 60)
    
    # Build α=0.95 features
    syn_norm = syn_features / (np.linalg.norm(syn_features, axis=1, keepdims=True) + 1e-8)
    emb_norm = emb_features / (np.linalg.norm(emb_features, axis=1, keepdims=True) + 1e-8)
    
    joint_95 = np.hstack([0.95 * emb_norm, 0.05 * syn_norm])
    joint_95 = joint_95 / (np.linalg.norm(joint_95, axis=1, keepdims=True) + 1e-8)
    
    # Cluster with different k values
    for k in [20, 30]:
        labels = hierarchical_clustering(joint_95, k, method="ward")
        print_cluster_analysis(joint_95, labels, lemmas, 
                              f"Hierarchical (Ward, k={k}, α=0.95)")
    
    # Save best results (convert numpy types)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj
    
    best_exp = experiments[0]
    results = convert_numpy({
        "experiments": experiments[:20],
        "best_config": best_exp,
        "recommendation": "Use α=0.95 for semantic clustering, α=0.5 for balanced",
    })
    
    with open(output_dir / "eng_enhanced_clustering.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\nResults saved to {output_dir / 'eng_enhanced_clustering.json'}")
    
    return results


# CLI interface
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"
    
    run_enhanced_clustering(
        output_dir=output_dir,
        syn_path=output_dir / "english_verb_features.npz",
        emb_path=output_dir / "english_verb_ctx_embeddings.npz",
        lemma_index_path=output_dir / "english_lemma_index.json",
        instances_path=output_dir / "english_verb_instances.jsonl",
    )

