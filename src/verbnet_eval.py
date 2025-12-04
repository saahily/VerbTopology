"""
VerbNet Evaluation for verb clustering.

Compares induced clusters against VerbNet classes using:
- V-Measure (homogeneity + completeness)
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
"""

from pathlib import Path
from typing import Optional
from collections import defaultdict
import json

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    v_measure_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)

import nltk
nltk.download('verbnet', quiet=True)
nltk.download('verbnet3', quiet=True)
from nltk.corpus import verbnet as vn


def load_verbnet_mapping() -> dict[str, list[str]]:
    """
    Load VerbNet lemma -> class mappings.
    
    Returns:
        Dict mapping lemma to list of VerbNet classes
    """
    lemma_to_classes = defaultdict(list)
    
    for classid in vn.classids():
        # Get lemmas for this class
        try:
            lemmas = vn.lemmas(classid)
            for lemma in lemmas:
                # Normalize lemma (remove underscores, lowercase)
                lemma_norm = lemma.replace('_', ' ').lower()
                # Also store the base form
                base = lemma_norm.split()[0] if ' ' in lemma_norm else lemma_norm
                lemma_to_classes[base].append(classid)
        except Exception:
            pass
    
    return dict(lemma_to_classes)


def get_primary_class(classes: list[str]) -> str:
    """Get the primary (most general) class for a verb."""
    if not classes:
        return 'UNKNOWN'
    # Sort by specificity (fewer dashes = more general)
    sorted_classes = sorted(classes, key=lambda x: x.count('-'))
    return sorted_classes[0]


def get_class_family(classid: str) -> str:
    """Get the top-level class family (e.g., 'admire-31.2-1' -> 'admire-31')."""
    parts = classid.split('-')
    if len(parts) >= 2:
        return f"{parts[0]}-{parts[1].split('.')[0]}"
    return classid


def evaluate_clustering(
    cluster_labels: list[int],
    lemmas: list[str],
    verbnet_map: dict[str, list[str]],
    use_family: bool = True,
) -> dict:
    """
    Evaluate clustering against VerbNet.
    
    Args:
        cluster_labels: Cluster assignment for each lemma
        lemmas: List of verb lemmas
        verbnet_map: VerbNet lemma -> classes mapping
        use_family: If True, use class families instead of specific classes
        
    Returns:
        Dict with evaluation metrics
    """
    # Find lemmas that appear in both our data and VerbNet
    matched_indices = []
    matched_cluster_labels = []
    matched_vn_labels = []
    
    # Build VerbNet class -> id mapping
    all_vn_classes = set()
    for classes in verbnet_map.values():
        for c in classes:
            if use_family:
                all_vn_classes.add(get_class_family(c))
            else:
                all_vn_classes.add(c)
    
    vn_class_to_id = {c: i for i, c in enumerate(sorted(all_vn_classes))}
    
    for i, lemma in enumerate(lemmas):
        if lemma in verbnet_map:
            classes = verbnet_map[lemma]
            primary = get_primary_class(classes)
            if use_family:
                primary = get_class_family(primary)
            
            matched_indices.append(i)
            matched_cluster_labels.append(cluster_labels[i])
            matched_vn_labels.append(vn_class_to_id[primary])
    
    if len(matched_indices) < 10:
        return {
            'coverage': len(matched_indices) / len(lemmas),
            'matched_count': len(matched_indices),
            'error': 'Too few matched lemmas for evaluation'
        }
    
    # Compute metrics
    pred = np.array(matched_cluster_labels)
    true = np.array(matched_vn_labels)
    
    return {
        'coverage': len(matched_indices) / len(lemmas),
        'matched_count': len(matched_indices),
        'v_measure': v_measure_score(true, pred),
        'homogeneity': homogeneity_score(true, pred),
        'completeness': completeness_score(true, pred),
        'ari': adjusted_rand_score(true, pred),
        'nmi': normalized_mutual_info_score(true, pred),
        'n_clusters_pred': len(set(pred)),
        'n_clusters_true': len(set(true)),
    }


def analyze_cluster_verbnet_overlap(
    cluster_labels: list[int],
    lemmas: list[str],
    verbnet_map: dict[str, list[str]],
    top_n: int = 10,
) -> list[dict]:
    """
    Analyze which VerbNet classes each cluster overlaps with.
    """
    # Group lemmas by cluster
    clusters = defaultdict(list)
    for i, label in enumerate(cluster_labels):
        clusters[label].append(lemmas[i])
    
    analysis = []
    
    for cluster_id, cluster_lemmas in sorted(clusters.items()):
        # Count VerbNet class occurrences
        vn_counts = defaultdict(int)
        matched = 0
        
        for lemma in cluster_lemmas:
            if lemma in verbnet_map:
                matched += 1
                for c in verbnet_map[lemma]:
                    family = get_class_family(c)
                    vn_counts[family] += 1
        
        # Get top VerbNet classes
        top_classes = sorted(vn_counts.items(), key=lambda x: -x[1])[:3]
        
        analysis.append({
            'cluster_id': cluster_id,
            'size': len(cluster_lemmas),
            'vn_coverage': matched / len(cluster_lemmas) if cluster_lemmas else 0,
            'top_vn_classes': top_classes,
            'sample_verbs': cluster_lemmas[:10],
        })
    
    return analysis


def run_verbnet_evaluation(
    output_dir: Path,
    k_values: list[int] = [10, 20, 30, 50],
):
    """
    Run VerbNet evaluation for English clustering.
    """
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("Phase 6: VerbNet Evaluation")
    print("=" * 60)
    
    # Load VerbNet
    print("\nLoading VerbNet...")
    verbnet_map = load_verbnet_mapping()
    print(f"  VerbNet lemmas: {len(verbnet_map)}")
    
    # Load features and lemmas
    print("\nLoading features...")
    joint = np.load(output_dir / 'english_joint_features.npz')['features']
    with open(output_dir / 'english_joint_lemma_index.json') as f:
        idx = json.load(f)
    lemmas = idx['index_to_lemma']
    
    # Check coverage
    matched = sum(1 for l in lemmas if l in verbnet_map)
    print(f"  Our lemmas: {len(lemmas)}")
    print(f"  Matched in VerbNet: {matched} ({100*matched/len(lemmas):.1f}%)")
    
    # Compute hierarchical clustering
    print("\nComputing hierarchical clustering...")
    Z = linkage(joint, method='ward')
    
    # Evaluate at different k values
    results = []
    
    print("\nEvaluating at different cluster counts...")
    print("-" * 60)
    print(f"{'k':>5} | {'V-Measure':>10} | {'ARI':>8} | {'NMI':>8} | {'Homog.':>8} | {'Compl.':>8}")
    print("-" * 60)
    
    for k in k_values:
        labels = fcluster(Z, t=k, criterion='maxclust')
        metrics = evaluate_clustering(labels.tolist(), lemmas, verbnet_map, use_family=True)
        
        if 'error' not in metrics:
            print(f"{k:>5} | {metrics['v_measure']:>10.3f} | {metrics['ari']:>8.3f} | "
                  f"{metrics['nmi']:>8.3f} | {metrics['homogeneity']:>8.3f} | {metrics['completeness']:>8.3f}")
            
            results.append({
                'k': k,
                **metrics
            })
    
    print("-" * 60)
    
    # Find best k
    if results:
        best = max(results, key=lambda x: x['v_measure'])
        print(f"\nBest V-Measure: k={best['k']} with V={best['v_measure']:.3f}")
        
        # Detailed analysis at best k
        print(f"\nDetailed cluster analysis at k={best['k']}...")
        labels = fcluster(Z, t=best['k'], criterion='maxclust')
        analysis = analyze_cluster_verbnet_overlap(labels.tolist(), lemmas, verbnet_map)
        
        # Show top clusters by VerbNet coherence
        print("\nClusters with highest VerbNet coherence:")
        print("-" * 60)
        
        coherent = sorted(analysis, key=lambda x: -x['vn_coverage'])[:10]
        for a in coherent:
            if a['top_vn_classes']:
                top_class = a['top_vn_classes'][0]
                print(f"\nCluster {a['cluster_id']} ({a['size']} verbs, {100*a['vn_coverage']:.0f}% VN coverage)")
                print(f"  Top VN class: {top_class[0]} ({top_class[1]} verbs)")
                print(f"  Sample verbs: {a['sample_verbs'][:8]}")
    
    # Save results
    results_path = output_dir / 'eng_verbnet_evaluation.json'
    with open(results_path, 'w') as f:
        json.dump({
            'verbnet_coverage': matched / len(lemmas),
            'matched_lemmas': matched,
            'total_lemmas': len(lemmas),
            'results_by_k': results,
        }, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Generate markdown report
    report_path = output_dir / 'eng_verbnet_evaluation.md'
    with open(report_path, 'w') as f:
        f.write("# VerbNet Evaluation Results\n\n")
        f.write(f"**Coverage:** {matched}/{len(lemmas)} lemmas ({100*matched/len(lemmas):.1f}%) found in VerbNet\n\n")
        f.write("## Metrics by Cluster Count\n\n")
        f.write("| k | V-Measure | ARI | NMI | Homogeneity | Completeness |\n")
        f.write("|---|-----------|-----|-----|-------------|-------------|\n")
        for r in results:
            f.write(f"| {r['k']} | {r['v_measure']:.3f} | {r['ari']:.3f} | "
                    f"{r['nmi']:.3f} | {r['homogeneity']:.3f} | {r['completeness']:.3f} |\n")
        
        if results:
            best = max(results, key=lambda x: x['v_measure'])
            f.write(f"\n**Best:** k={best['k']} with V-Measure={best['v_measure']:.3f}\n")
    
    print(f"Report saved to {report_path}")
    
    return results


# Random baseline
def random_baseline(n_samples: int, n_clusters: int, n_trials: int = 100) -> dict:
    """Compute random baseline for comparison."""
    from sklearn.metrics import v_measure_score
    
    scores = []
    for _ in range(n_trials):
        pred = np.random.randint(0, n_clusters, n_samples)
        true = np.random.randint(0, n_clusters, n_samples)
        scores.append(v_measure_score(true, pred))
    
    return {
        'mean': np.mean(scores),
        'std': np.std(scores),
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "outputs"
    
    run_verbnet_evaluation(output_dir)

