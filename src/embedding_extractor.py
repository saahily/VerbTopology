"""
Contextual embedding extraction for verb instances.

Uses XLM-RoBERTa to extract contextualized embeddings for each verb instance,
enabling semantic-aware clustering and cross-lingual comparison.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import json
import warnings

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from verb_extractor import VerbInstance, load_verb_instances


@dataclass
class EmbeddingConfig:
    """Configuration for embedding extraction."""
    model_name: str = "xlm-roberta-base"  # or "xlm-roberta-large"
    layer: int = -1  # Which layer to extract (-1 = last, -4 = mean of last 4)
    pooling: str = "mean"  # How to pool subword tokens: "mean", "first", "last"
    batch_size: int = 32
    max_length: int = 512
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    
    def get_device(self) -> torch.device:
        """Determine the best available device."""
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(self.device)


class VerbEmbeddingExtractor:
    """
    Extracts contextualized embeddings for verb instances using XLM-R.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the extractor.
        
        Args:
            config: Embedding configuration (uses defaults if None)
        """
        self.config = config or EmbeddingConfig()
        self.device = self.config.get_device()
        
        print(f"Loading model: {self.config.model_name}")
        print(f"Device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModel.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def _find_verb_token_indices(
        self,
        sentence: str,
        verb_form: str,
        encoding,
    ) -> list[int]:
        """
        Find the token indices corresponding to the verb in the tokenized sentence.
        
        Returns list of subword token indices for the verb.
        """
        # Tokenize the verb form separately to understand its subword structure
        verb_tokens = self.tokenizer.tokenize(verb_form)
        num_verb_tokens = len(verb_tokens)
        
        # Get all tokens (excluding special tokens)
        all_tokens = self.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])
        
        # Find the verb tokens in the sequence
        # Strategy: sliding window match
        for i in range(1, len(all_tokens) - num_verb_tokens):  # Skip [CLS] and [SEP]
            window = all_tokens[i:i + num_verb_tokens]
            
            # Check for match (handle XLM-R's subword prefix marker '▁')
            if self._tokens_match(window, verb_tokens):
                return list(range(i, i + num_verb_tokens))
        
        # Fallback: try to find partial match
        for i, token in enumerate(all_tokens[1:-1], start=1):
            clean_token = token.replace('▁', '').lower()
            clean_verb = verb_form.lower()
            if clean_verb.startswith(clean_token) or clean_token.startswith(clean_verb[:3]):
                # Found approximate match, return single token
                return [i]
        
        # Last resort: return middle token (not ideal but prevents crash)
        mid = len(all_tokens) // 2
        return [mid]
    
    def _tokens_match(self, window: list[str], target: list[str]) -> bool:
        """Check if token window matches target (handling subword markers)."""
        if len(window) != len(target):
            return False
        
        for w, t in zip(window, target):
            # Normalize: remove subword marker and lowercase
            w_clean = w.replace('▁', '').lower()
            t_clean = t.replace('▁', '').lower()
            if w_clean != t_clean:
                return False
        return True
    
    def _pool_subword_embeddings(
        self,
        hidden_states: torch.Tensor,
        token_indices: list[int],
    ) -> np.ndarray:
        """
        Pool embeddings for subword tokens into a single vector.
        
        Args:
            hidden_states: Shape (1, seq_len, hidden_dim)
            token_indices: Indices of verb subword tokens
            
        Returns:
            Pooled embedding vector
        """
        # Extract embeddings for verb tokens
        verb_embeddings = hidden_states[0, token_indices, :]  # (num_subwords, hidden_dim)
        
        if self.config.pooling == "mean":
            pooled = verb_embeddings.mean(dim=0)
        elif self.config.pooling == "first":
            pooled = verb_embeddings[0]
        elif self.config.pooling == "last":
            pooled = verb_embeddings[-1]
        else:
            raise ValueError(f"Unknown pooling method: {self.config.pooling}")
        
        return pooled.cpu().numpy()
    
    def extract_single(
        self,
        sentence: str,
        verb_form: str,
    ) -> np.ndarray:
        """
        Extract embedding for a single verb instance.
        
        Args:
            sentence: Full sentence text
            verb_form: Surface form of the verb
            
        Returns:
            Embedding vector of shape (hidden_dim,)
        """
        # Tokenize
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )
        
        # Find verb token indices (before moving to device)
        token_indices = self._find_verb_token_indices(sentence, verb_form, encoding)
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**encoding, output_hidden_states=True)
        
        # Extract from specified layer
        if self.config.layer == -1:
            hidden_states = outputs.last_hidden_state
        elif self.config.layer < 0:
            # Mean of last N layers
            num_layers = abs(self.config.layer)
            hidden_layers = outputs.hidden_states[-num_layers:]
            hidden_states = torch.stack(hidden_layers).mean(dim=0)
        else:
            hidden_states = outputs.hidden_states[self.config.layer]
        
        # Pool subword embeddings
        return self._pool_subword_embeddings(hidden_states, token_indices)
    
    def extract_batch(
        self,
        instances: list[VerbInstance],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of verb instances.
        
        Args:
            instances: List of verb instances
            show_progress: Whether to show progress bar
            
        Returns:
            Embedding matrix of shape (num_instances, hidden_dim)
        """
        embeddings = []
        n = len(instances)
        
        for i, inst in enumerate(instances):
            if show_progress and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n} instances...")
            
            if inst.sentence_text:
                emb = self.extract_single(inst.sentence_text, inst.form)
            else:
                # Fallback: use verb form alone (not ideal)
                emb = self.extract_single(inst.form, inst.form)
            
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)


def aggregate_to_lemma_level(
    instances: list[VerbInstance],
    instance_embeddings: np.ndarray,
    min_instances: int = 3,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    """
    Aggregate instance embeddings to lemma level via mean pooling.
    
    Args:
        instances: List of verb instances
        instance_embeddings: Instance-level embeddings (n_instances, hidden_dim)
        min_instances: Minimum instances for a lemma to be included
        
    Returns:
        - Lemma embeddings (n_lemmas, hidden_dim)
        - List of lemma names
        - Dict of lemma -> instance count
    """
    from collections import defaultdict
    
    # Group by lemma
    lemma_indices: dict[str, list[int]] = defaultdict(list)
    for i, inst in enumerate(instances):
        lemma_indices[inst.lemma].append(i)
    
    # Filter by minimum count and sort by frequency
    filtered = {
        lemma: indices 
        for lemma, indices in lemma_indices.items() 
        if len(indices) >= min_instances
    }
    sorted_lemmas = sorted(filtered.keys(), key=lambda x: -len(filtered[x]))
    
    # Compute mean embeddings
    hidden_dim = instance_embeddings.shape[1]
    lemma_embeddings = np.zeros((len(sorted_lemmas), hidden_dim), dtype=np.float32)
    instance_counts = {}
    
    for i, lemma in enumerate(sorted_lemmas):
        indices = filtered[lemma]
        lemma_embeddings[i] = instance_embeddings[indices].mean(axis=0)
        instance_counts[lemma] = len(indices)
    
    return lemma_embeddings, sorted_lemmas, instance_counts


def build_joint_features(
    lemma_embeddings: np.ndarray,
    syntactic_features: np.ndarray,
    embedding_lemmas: list[str],
    syntactic_lemmas: list[str],
    alpha: float = 0.95,  # Updated: heavily weight embeddings for semantic clustering
) -> tuple[np.ndarray, list[str]]:
    """
    Build joint feature matrix combining embeddings and syntactic features.
    
    Args:
        lemma_embeddings: Embedding matrix (n_emb_lemmas, hidden_dim)
        syntactic_features: Syntactic feature matrix (n_syn_lemmas, n_features)
        embedding_lemmas: Lemma list for embeddings
        syntactic_lemmas: Lemma list for syntactic features
        alpha: Weight for embeddings (1-alpha for syntactic)
        
    Returns:
        - Joint feature matrix (n_common_lemmas, hidden_dim + n_features)
        - List of common lemmas
    """
    # Find common lemmas
    emb_set = set(embedding_lemmas)
    syn_set = set(syntactic_lemmas)
    common = emb_set & syn_set
    
    # Sort by embedding order (frequency)
    common_lemmas = [l for l in embedding_lemmas if l in common]
    
    # Build index mappings
    emb_idx = {l: i for i, l in enumerate(embedding_lemmas)}
    syn_idx = {l: i for i, l in enumerate(syntactic_lemmas)}
    
    # L2 normalize both feature sets
    emb_norm = lemma_embeddings / (np.linalg.norm(lemma_embeddings, axis=1, keepdims=True) + 1e-8)
    syn_norm = syntactic_features / (np.linalg.norm(syntactic_features, axis=1, keepdims=True) + 1e-8)
    
    # Build joint matrix
    hidden_dim = lemma_embeddings.shape[1]
    n_syn_features = syntactic_features.shape[1]
    joint = np.zeros((len(common_lemmas), hidden_dim + n_syn_features), dtype=np.float32)
    
    for i, lemma in enumerate(common_lemmas):
        emb_vec = emb_norm[emb_idx[lemma]]
        syn_vec = syn_norm[syn_idx[lemma]]
        
        # Weighted concatenation
        joint[i, :hidden_dim] = alpha * emb_vec
        joint[i, hidden_dim:] = (1 - alpha) * syn_vec
    
    # Re-normalize the joint vector
    joint = joint / (np.linalg.norm(joint, axis=1, keepdims=True) + 1e-8)
    
    return joint, common_lemmas


def extract_language_embeddings(
    instances: list[VerbInstance],
    language: str,
    output_dir: Path,
    syntactic_features_path: Optional[Path] = None,
    syntactic_lemma_index_path: Optional[Path] = None,
    config: Optional[EmbeddingConfig] = None,
    min_instances: int = 3,
    alpha: float = 0.7,
) -> dict:
    """
    Full embedding extraction pipeline for a language.
    
    Args:
        instances: List of verb instances
        language: Language code
        output_dir: Output directory
        syntactic_features_path: Path to syntactic features .npz (for joint features)
        syntactic_lemma_index_path: Path to syntactic lemma index .json
        config: Embedding configuration
        min_instances: Minimum instances per lemma
        alpha: Weight for embeddings in joint features
        
    Returns:
        Dictionary with extraction statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Extracting embeddings for {language}")
    print(f"{'='*60}")
    print(f"Total instances: {len(instances):,}")
    
    # Initialize extractor
    extractor = VerbEmbeddingExtractor(config)
    
    # Extract instance-level embeddings
    print("\nExtracting instance embeddings...")
    instance_embeddings = extractor.extract_batch(instances, show_progress=True)
    print(f"Instance embeddings shape: {instance_embeddings.shape}")
    
    # Aggregate to lemma level
    print("\nAggregating to lemma level...")
    lemma_embeddings, lemma_list, instance_counts = aggregate_to_lemma_level(
        instances, instance_embeddings, min_instances=min_instances
    )
    print(f"Lemma embeddings shape: {lemma_embeddings.shape}")
    print(f"Lemmas (≥{min_instances} instances): {len(lemma_list)}")
    
    # Save lemma embeddings
    np.savez_compressed(
        output_dir / f"{language}_verb_ctx_embeddings.npz",
        embeddings=lemma_embeddings,
    )
    
    # Save lemma index
    with open(output_dir / f"{language}_embedding_lemma_index.json", "w") as f:
        json.dump({
            "lemma_to_index": {l: i for i, l in enumerate(lemma_list)},
            "index_to_lemma": lemma_list,
            "instance_counts": instance_counts,
        }, f, indent=2, ensure_ascii=False)
    
    # Build joint features if syntactic features available
    joint_lemmas = None
    if syntactic_features_path and syntactic_lemma_index_path:
        print("\nBuilding joint features...")
        
        syn_features = np.load(syntactic_features_path)["features"]
        with open(syntactic_lemma_index_path) as f:
            syn_index = json.load(f)
        syn_lemmas = syn_index["index_to_lemma"]
        
        joint_features, joint_lemmas = build_joint_features(
            lemma_embeddings, syn_features,
            lemma_list, syn_lemmas,
            alpha=alpha,
        )
        
        print(f"Joint features shape: {joint_features.shape}")
        print(f"Common lemmas: {len(joint_lemmas)}")
        
        # Save joint features
        np.savez_compressed(
            output_dir / f"{language}_joint_features.npz",
            features=joint_features,
        )
        
        with open(output_dir / f"{language}_joint_lemma_index.json", "w") as f:
            json.dump({
                "lemma_to_index": {l: i for i, l in enumerate(joint_lemmas)},
                "index_to_lemma": joint_lemmas,
            }, f, indent=2, ensure_ascii=False)
    
    stats = {
        "language": language,
        "total_instances": len(instances),
        "instance_embedding_shape": list(instance_embeddings.shape),
        "lemma_embedding_shape": list(lemma_embeddings.shape),
        "num_lemmas": len(lemma_list),
        "embedding_dim": extractor.embedding_dim,
        "model": extractor.config.model_name,
        "top_lemmas": lemma_list[:10],
    }
    
    if joint_lemmas:
        stats["joint_feature_dim"] = lemma_embeddings.shape[1] + syn_features.shape[1]
        stats["joint_lemmas"] = len(joint_lemmas)
    
    print(f"\nSaved to {output_dir}")
    
    return stats


def sanity_check_embeddings(
    embeddings: np.ndarray,
    lemmas: list[str],
    pairs: list[tuple[str, str]],
) -> None:
    """
    Sanity check: verify that semantically similar verbs have higher cosine similarity.
    
    Args:
        embeddings: Lemma embedding matrix
        lemmas: Lemma list
        pairs: List of (verb1, verb2) pairs expected to be similar
    """
    from scipy.spatial.distance import cosine
    
    lemma_idx = {l: i for i, l in enumerate(lemmas)}
    
    print("\nSanity Check: Semantic Similarity")
    print("-" * 50)
    
    for v1, v2 in pairs:
        if v1 in lemma_idx and v2 in lemma_idx:
            e1 = embeddings[lemma_idx[v1]]
            e2 = embeddings[lemma_idx[v2]]
            sim = 1 - cosine(e1, e2)
            print(f"  {v1} <-> {v2}: {sim:.3f}")
        else:
            missing = [v for v in (v1, v2) if v not in lemma_idx]
            print(f"  {v1} <-> {v2}: SKIPPED (missing: {missing})")


def run_embedding_extraction(
    input_dir: Path,
    output_dir: Path,
    min_instances: int = 3,
    model_name: str = "xlm-roberta-base",
) -> dict:
    """
    Run embedding extraction for all languages.
    
    Args:
        input_dir: Directory with verb instance JSONL files
        output_dir: Output directory
        min_instances: Minimum instances per lemma
        model_name: Transformer model to use
        
    Returns:
        Dictionary of stats per language
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    config = EmbeddingConfig(model_name=model_name)
    
    # Find all verb instance files
    instance_files = list(input_dir.glob("*_verb_instances.jsonl"))
    
    print(f"Found {len(instance_files)} language(s) to process")
    
    all_stats = {}
    
    for filepath in sorted(instance_files):
        language = filepath.stem.replace("_verb_instances", "")
        
        # Load instances
        instances = load_verb_instances(filepath)
        
        # Check for syntactic features
        syn_path = output_dir / f"{language}_verb_features.npz"
        syn_idx_path = output_dir / f"{language}_lemma_index.json"
        
        if syn_path.exists() and syn_idx_path.exists():
            syn_features_path = syn_path
            syn_lemma_path = syn_idx_path
        else:
            syn_features_path = None
            syn_lemma_path = None
            print(f"  Note: Syntactic features not found for {language}, skipping joint features")
        
        stats = extract_language_embeddings(
            instances,
            language,
            output_dir,
            syntactic_features_path=syn_features_path,
            syntactic_lemma_index_path=syn_lemma_path,
            config=config,
            min_instances=min_instances,
        )
        
        all_stats[language] = stats
    
    # Save combined stats
    with open(output_dir / "embedding_extraction_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Summary written to: {output_dir / 'embedding_extraction_stats.json'}")
    
    return all_stats


# CLI interface
if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent.parent
    input_dir = project_root / "outputs"
    output_dir = project_root / "outputs"
    
    print("=" * 60)
    print("VerbTopology: Contextual Embedding Extraction (Phase 3b)")
    print("=" * 60)
    
    # Run extraction
    stats = run_embedding_extraction(input_dir, output_dir, min_instances=3)
    
    # Sanity check for English
    print("\n" + "=" * 60)
    print("Sanity Check: English Verb Similarities")
    print("=" * 60)
    
    eng_emb_path = output_dir / "english_verb_ctx_embeddings.npz"
    eng_idx_path = output_dir / "english_embedding_lemma_index.json"
    
    if eng_emb_path.exists():
        embeddings = np.load(eng_emb_path)["embeddings"]
        with open(eng_idx_path) as f:
            idx = json.load(f)
        lemmas = idx["index_to_lemma"]
        
        # Test pairs (semantically similar verbs)
        similar_pairs = [
            ("give", "donate"),
            ("run", "sprint"),
            ("say", "tell"),
            ("think", "believe"),
            ("walk", "run"),
            ("buy", "purchase"),
            ("see", "watch"),
            ("eat", "consume"),
        ]
        
        # Test dissimilar pairs
        dissimilar_pairs = [
            ("give", "sleep"),
            ("run", "think"),
            ("say", "eat"),
        ]
        
        print("\nExpected HIGH similarity:")
        sanity_check_embeddings(embeddings, lemmas, similar_pairs)
        
        print("\nExpected LOW similarity:")
        sanity_check_embeddings(embeddings, lemmas, dissimilar_pairs)
