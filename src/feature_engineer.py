"""
Syntactic frame feature engineering.

Converts verb instances into feature vectors suitable for clustering.
Features are aggregated at the lemma level (each verb lemma gets one vector).
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import numpy as np
from scipy import sparse

from verb_extractor import VerbInstance, load_verb_instances


@dataclass
class FeatureSchema:
    """
    Defines the feature space for a language.
    
    Features are organized into groups:
    - Core argument configuration (binary: has_nsubj, has_obj, etc.)
    - Oblique subcategorization (binary: obl_case_X, obl_adp_X)
    - Transitivity pattern (one-hot)
    - Voice (one-hot, if morphologically marked)
    """
    
    # Feature names in order
    feature_names: list[str] = field(default_factory=list)
    
    # Indices for feature groups (for analysis)
    core_arg_indices: list[int] = field(default_factory=list)
    oblique_indices: list[int] = field(default_factory=list)
    transitivity_indices: list[int] = field(default_factory=list)
    voice_indices: list[int] = field(default_factory=list)
    
    @property
    def num_features(self) -> int:
        return len(self.feature_names)
    
    def get_index(self, name: str) -> int:
        return self.feature_names.index(name)
    
    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "num_features": self.num_features,
            "groups": {
                "core_arg": self.core_arg_indices,
                "oblique": self.oblique_indices,
                "transitivity": self.transitivity_indices,
                "voice": self.voice_indices,
            }
        }


class FeatureExtractor:
    """
    Extracts features from verb instances and aggregates at lemma level.
    """
    
    # Core argument features (present in all languages)
    CORE_ARG_FEATURES = [
        "has_nsubj",      # Has nominal subject
        "has_nsubj_pass", # Has passive subject
        "has_csubj",      # Has clausal subject
        "has_obj",        # Has direct object
        "has_iobj",       # Has indirect object
        "has_obl",        # Has any oblique
        "has_xcomp",      # Has open clausal complement
        "has_ccomp",      # Has closed clausal complement
    ]
    
    # Transitivity categories
    TRANSITIVITY_CATEGORIES = [
        "trans_intransitive",
        "trans_transitive", 
        "trans_ditransitive",
        "trans_impersonal",
    ]
    
    # Voice categories (language-dependent)
    VOICE_CATEGORIES = [
        "voice_Act",   # Active
        "voice_Pass",  # Passive
        "voice_Mid",   # Middle
    ]
    
    def __init__(
        self, 
        min_instance_count: int = 3,
        include_voice: bool = True,
        max_oblique_features: int = 30,
    ):
        """
        Initialize feature extractor.
        
        Args:
            min_instance_count: Minimum instances for a lemma to be included
            include_voice: Whether to include voice features
            max_oblique_features: Maximum number of oblique subcategorization features
        """
        self.min_instance_count = min_instance_count
        self.include_voice = include_voice
        self.max_oblique_features = max_oblique_features
    
    def _get_oblique_key(self, arg: dict) -> Optional[str]:
        """
        Get oblique subcategorization key from argument.
        
        Priority: adposition > case > generic 'obl'
        """
        if not arg["deprel"].startswith("obl"):
            return None
        
        # Try adposition first (most specific for English)
        if arg.get("adposition"):
            return f"obl_adp_{arg['adposition'].lower()}"
        
        # Try case (for Sanskrit, Telugu, etc.)
        if arg.get("case"):
            return f"obl_case_{arg['case']}"
        
        # Subtype from deprel (e.g., obl:tmod -> obl_tmod)
        if ":" in arg["deprel"]:
            subtype = arg["deprel"].split(":")[1]
            return f"obl_{subtype}"
        
        return None
    
    def _collect_oblique_vocabulary(
        self, 
        instances: list[VerbInstance]
    ) -> list[str]:
        """
        Collect most frequent oblique subcategorization features.
        """
        obl_counts: dict[str, int] = defaultdict(int)
        
        for inst in instances:
            for arg in inst.arguments:
                key = self._get_oblique_key(arg.to_dict() if hasattr(arg, 'to_dict') else arg)
                if key:
                    obl_counts[key] += 1
        
        # Sort by frequency and take top N
        sorted_obls = sorted(obl_counts.items(), key=lambda x: -x[1])
        top_obls = [k for k, v in sorted_obls[:self.max_oblique_features] if v >= 5]
        
        return sorted(top_obls)
    
    def _check_voice_presence(self, instances: list[VerbInstance]) -> bool:
        """Check if voice features are present in the data."""
        for inst in instances:
            if inst.voice:
                return True
        return False
    
    def build_schema(self, instances: list[VerbInstance]) -> FeatureSchema:
        """
        Build feature schema from instances.
        
        Dynamically determines oblique features based on what's in the data.
        """
        schema = FeatureSchema()
        idx = 0
        
        # Core argument features
        for feat in self.CORE_ARG_FEATURES:
            schema.feature_names.append(feat)
            schema.core_arg_indices.append(idx)
            idx += 1
        
        # Oblique subcategorization (data-driven)
        oblique_vocab = self._collect_oblique_vocabulary(instances)
        for feat in oblique_vocab:
            schema.feature_names.append(feat)
            schema.oblique_indices.append(idx)
            idx += 1
        
        # Transitivity
        for feat in self.TRANSITIVITY_CATEGORIES:
            schema.feature_names.append(feat)
            schema.transitivity_indices.append(idx)
            idx += 1
        
        # Voice (only if present in data)
        if self.include_voice and self._check_voice_presence(instances):
            for feat in self.VOICE_CATEGORIES:
                schema.feature_names.append(feat)
                schema.voice_indices.append(idx)
                idx += 1
        
        return schema
    
    def extract_instance_features(
        self, 
        instance: VerbInstance, 
        schema: FeatureSchema
    ) -> dict[str, float]:
        """
        Extract binary features from a single instance.
        
        Returns dict of feature_name -> 1.0 for present features.
        """
        features = {}
        
        # Core argument features
        deprels = {arg.deprel for arg in instance.arguments}
        base_deprels = {d.split(":")[0] for d in deprels}
        
        if "nsubj" in base_deprels or any(d.startswith("nsubj") and "pass" not in d for d in deprels):
            features["has_nsubj"] = 1.0
        if "nsubj:pass" in deprels:
            features["has_nsubj_pass"] = 1.0
        if "csubj" in base_deprels:
            features["has_csubj"] = 1.0
        if "obj" in deprels:
            features["has_obj"] = 1.0
        if "iobj" in deprels:
            features["has_iobj"] = 1.0
        if "obl" in base_deprels:
            features["has_obl"] = 1.0
        if "xcomp" in deprels:
            features["has_xcomp"] = 1.0
        if "ccomp" in deprels:
            features["has_ccomp"] = 1.0
        
        # Oblique subcategorization
        for arg in instance.arguments:
            key = self._get_oblique_key(arg.to_dict() if hasattr(arg, 'to_dict') else arg)
            if key and key in schema.feature_names:
                features[key] = 1.0
        
        # Transitivity
        trans = instance.transitivity_pattern
        trans_key = f"trans_{trans}"
        if trans_key in schema.feature_names:
            features[trans_key] = 1.0
        
        # Voice
        if instance.voice and self.include_voice:
            voice_key = f"voice_{instance.voice}"
            if voice_key in schema.feature_names:
                features[voice_key] = 1.0
        
        return features
    
    def aggregate_lemma_features(
        self,
        instances: list[VerbInstance],
        schema: FeatureSchema,
    ) -> tuple[np.ndarray, list[str], dict[str, int]]:
        """
        Aggregate features at the lemma level.
        
        For each lemma, computes the proportion of instances having each feature.
        
        Args:
            instances: List of verb instances
            schema: Feature schema
            
        Returns:
            - Feature matrix (num_lemmas x num_features)
            - List of lemma names (row index)
            - Dict of lemma -> instance count
        """
        # Group instances by lemma
        lemma_instances: dict[str, list[VerbInstance]] = defaultdict(list)
        for inst in instances:
            lemma_instances[inst.lemma].append(inst)
        
        # Filter by minimum count
        filtered_lemmas = {
            lemma: insts 
            for lemma, insts in lemma_instances.items() 
            if len(insts) >= self.min_instance_count
        }
        
        # Sort lemmas by frequency (most frequent first)
        sorted_lemmas = sorted(
            filtered_lemmas.keys(),
            key=lambda x: -len(filtered_lemmas[x])
        )
        
        # Build feature matrix
        num_lemmas = len(sorted_lemmas)
        num_features = schema.num_features
        
        feature_matrix = np.zeros((num_lemmas, num_features), dtype=np.float32)
        instance_counts = {}
        
        for row_idx, lemma in enumerate(sorted_lemmas):
            insts = filtered_lemmas[lemma]
            instance_counts[lemma] = len(insts)
            
            # Accumulate features across instances
            feature_sums = np.zeros(num_features, dtype=np.float32)
            
            for inst in insts:
                inst_features = self.extract_instance_features(inst, schema)
                for feat_name, value in inst_features.items():
                    feat_idx = schema.get_index(feat_name)
                    feature_sums[feat_idx] += value
            
            # Convert to proportions
            feature_matrix[row_idx] = feature_sums / len(insts)
        
        return feature_matrix, sorted_lemmas, instance_counts


def normalize_features(
    features: np.ndarray, 
    method: str = "l2"
) -> np.ndarray:
    """
    Normalize feature vectors.
    
    Args:
        features: Feature matrix (num_samples x num_features)
        method: 'l2' for L2 normalization, 'zscore' for standardization
        
    Returns:
        Normalized feature matrix
    """
    if method == "l2":
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return features / norms
    
    elif method == "zscore":
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (features - mean) / std
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


@dataclass
class FeatureResult:
    """Result of feature engineering for a language."""
    language: str
    num_lemmas: int
    num_features: int
    schema: FeatureSchema
    lemma_list: list[str]
    instance_counts: dict[str, int]
    
    # Feature statistics
    feature_coverage: dict[str, float]  # % of lemmas with non-zero value
    
    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "num_lemmas": self.num_lemmas,
            "num_features": self.num_features,
            "schema": self.schema.to_dict(),
            "lemma_count": len(self.lemma_list),
            "feature_coverage": self.feature_coverage,
            "top_lemmas": self.lemma_list[:20],
            "instance_counts": {k: self.instance_counts[k] for k in self.lemma_list[:20]},
        }


def engineer_features(
    instances: list[VerbInstance],
    language: str,
    min_instances: int = 3,
    normalize: str = "l2",
) -> tuple[np.ndarray, FeatureResult]:
    """
    Full feature engineering pipeline for a language.
    
    Args:
        instances: List of verb instances
        language: Language code
        min_instances: Minimum instances per lemma
        normalize: Normalization method ('l2', 'zscore', or None)
        
    Returns:
        - Feature matrix (normalized)
        - FeatureResult with metadata
    """
    extractor = FeatureExtractor(min_instance_count=min_instances)
    
    # Build schema
    schema = extractor.build_schema(instances)
    
    # Aggregate features
    features, lemmas, counts = extractor.aggregate_lemma_features(instances, schema)
    
    # Compute coverage statistics
    coverage = {}
    for i, feat_name in enumerate(schema.feature_names):
        nonzero = np.sum(features[:, i] > 0)
        coverage[feat_name] = float(nonzero) / len(lemmas) if lemmas else 0
    
    # Normalize
    if normalize:
        features = normalize_features(features, method=normalize)
    
    result = FeatureResult(
        language=language,
        num_lemmas=len(lemmas),
        num_features=schema.num_features,
        schema=schema,
        lemma_list=lemmas,
        instance_counts=counts,
        feature_coverage=coverage,
    )
    
    return features, result


def save_features(
    features: np.ndarray,
    result: FeatureResult,
    output_dir: Path,
    language: str,
):
    """Save feature matrix and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature matrix
    np.savez_compressed(
        output_dir / f"{language}_verb_features.npz",
        features=features,
    )
    
    # Save schema/feature names
    with open(output_dir / f"{language}_feature_names.json", "w") as f:
        json.dump(result.schema.to_dict(), f, indent=2)
    
    # Save lemma index
    lemma_index = {lemma: i for i, lemma in enumerate(result.lemma_list)}
    with open(output_dir / f"{language}_lemma_index.json", "w") as f:
        json.dump({
            "lemma_to_index": lemma_index,
            "index_to_lemma": result.lemma_list,
            "instance_counts": result.instance_counts,
        }, f, indent=2, ensure_ascii=False)


def load_features(
    output_dir: Path,
    language: str,
) -> tuple[np.ndarray, dict, dict]:
    """
    Load saved features.
    
    Returns:
        - Feature matrix
        - Feature schema dict
        - Lemma index dict
    """
    output_dir = Path(output_dir)
    
    features = np.load(output_dir / f"{language}_verb_features.npz")["features"]
    
    with open(output_dir / f"{language}_feature_names.json") as f:
        schema = json.load(f)
    
    with open(output_dir / f"{language}_lemma_index.json") as f:
        lemma_index = json.load(f)
    
    return features, schema, lemma_index


def run_feature_engineering(
    input_dir: Path,
    output_dir: Path,
    min_instances: int = 3,
) -> dict:
    """
    Run feature engineering for all languages.
    
    Args:
        input_dir: Directory with verb instance JSONL files
        output_dir: Output directory
        min_instances: Minimum instances per lemma
        
    Returns:
        Dictionary of results per language
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Find all verb instance files
    instance_files = list(input_dir.glob("*_verb_instances.jsonl"))
    
    print(f"Found {len(instance_files)} language(s) to process")
    print()
    
    all_results = {}
    
    for filepath in sorted(instance_files):
        language = filepath.stem.replace("_verb_instances", "")
        print(f"Processing {language}...")
        
        # Load instances
        instances = load_verb_instances(filepath)
        print(f"  Loaded {len(instances):,} verb instances")
        
        # Engineer features
        features, result = engineer_features(
            instances, 
            language,
            min_instances=min_instances,
        )
        
        print(f"  Lemmas (≥{min_instances} instances): {result.num_lemmas:,}")
        print(f"  Features: {result.num_features}")
        print(f"  Feature matrix shape: {features.shape}")
        
        # Show feature groups
        schema = result.schema
        print(f"  Feature groups:")
        print(f"    - Core argument: {len(schema.core_arg_indices)} features")
        print(f"    - Oblique subcat: {len(schema.oblique_indices)} features")
        print(f"    - Transitivity: {len(schema.transitivity_indices)} features")
        print(f"    - Voice: {len(schema.voice_indices)} features")
        
        # Show top coverage features
        top_coverage = sorted(result.feature_coverage.items(), key=lambda x: -x[1])[:5]
        print(f"  Top features by coverage:")
        for feat, cov in top_coverage:
            print(f"    - {feat}: {cov*100:.1f}%")
        
        # Save
        save_features(features, result, output_dir, language)
        print(f"  Saved to {output_dir}")
        print()
        
        all_results[language] = result.to_dict()
    
    # Save combined summary
    with open(output_dir / "feature_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Summary written to: {output_dir / 'feature_summary.json'}")
    
    return all_results


# CLI interface
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "outputs"
    output_dir = project_root / "outputs"
    
    print("=" * 60)
    print("VerbTopology: Syntactic Frame Feature Engineering")
    print("=" * 60)
    print()
    
    results = run_feature_engineering(input_dir, output_dir, min_instances=3)
    
    # Quick sanity check on English
    print()
    print("=" * 60)
    print("Sanity Check: English High-Frequency Verbs")
    print("=" * 60)
    
    features, schema, lemma_idx = load_features(output_dir, "english")
    lemmas = lemma_idx["index_to_lemma"]
    feat_names = schema["feature_names"]
    
    # Check a few verbs
    check_verbs = ["give", "put", "go", "see", "think", "say"]
    
    for verb in check_verbs:
        if verb in lemma_idx["lemma_to_index"]:
            idx = lemma_idx["lemma_to_index"][verb]
            vec = features[idx]
            
            print(f"\n{verb} (n={lemma_idx['instance_counts'][verb]}):")
            
            # Show features with value > 0.3
            strong_feats = [(feat_names[i], vec[i]) for i in range(len(vec)) if vec[i] > 0.3]
            strong_feats.sort(key=lambda x: -x[1])
            
            for feat, val in strong_feats[:8]:
                print(f"  {feat}: {val:.2f}")

