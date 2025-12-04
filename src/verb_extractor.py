"""
Verb and argument extraction pipeline.

Extracts all verb instances with their core arguments from UD treebanks,
creating the base data for syntactic frame feature engineering.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
import json
import random

from data_loader import TreebankLoader, Sentence, Token


@dataclass
class Argument:
    """Represents a syntactic argument of a verb."""
    deprel: str                    # Dependency relation (nsubj, obj, obl, etc.)
    form: str                      # Surface form
    lemma: str                     # Lemma (may be "_" for Telugu)
    upos: str                      # Part of speech
    case: Optional[str] = None     # Case feature (if present)
    adposition: Optional[str] = None  # Governing adposition (for obl)
    head_position: int = 0         # Position relative to verb (-1=before, 1=after)
    
    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class VerbInstance:
    """
    Represents a single verb occurrence with its arguments.
    
    This is the fundamental unit for feature engineering:
    each verb instance captures how a verb is used in context.
    """
    # Verb identification
    lemma: str                     # Verb lemma (or form for Telugu)
    form: str                      # Surface form
    verb_id: int                   # Token ID in sentence
    
    # Morphological features
    voice: Optional[str] = None    # Voice (Act, Pass, Mid)
    tense: Optional[str] = None    # Tense
    aspect: Optional[str] = None   # Aspect
    mood: Optional[str] = None     # Mood
    person: Optional[str] = None   # Person
    number: Optional[str] = None   # Number
    
    # Arguments
    arguments: list[Argument] = field(default_factory=list)
    
    # Context
    sent_id: Optional[str] = None  # Sentence identifier
    sentence_text: Optional[str] = None  # Full sentence text
    language: str = ""             # Source language
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "lemma": self.lemma,
            "form": self.form,
            "verb_id": self.verb_id,
            "language": self.language,
            "arguments": [arg.to_dict() for arg in self.arguments],
        }
        
        # Add optional fields if present
        for field_name in ["voice", "tense", "aspect", "mood", "person", "number", 
                          "sent_id", "sentence_text"]:
            value = getattr(self, field_name)
            if value is not None:
                d[field_name] = value
        
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "VerbInstance":
        """Create from dictionary."""
        args = [Argument(**a) for a in d.pop("arguments", [])]
        return cls(**d, arguments=args)
    
    # Derived properties for analysis
    
    @property
    def has_subject(self) -> bool:
        return any(a.deprel in ("nsubj", "nsubj:pass", "csubj") for a in self.arguments)
    
    @property
    def has_object(self) -> bool:
        return any(a.deprel in ("obj", "iobj") for a in self.arguments)
    
    @property
    def has_oblique(self) -> bool:
        return any(a.deprel.startswith("obl") for a in self.arguments)
    
    @property
    def has_clausal_complement(self) -> bool:
        return any(a.deprel in ("xcomp", "ccomp") for a in self.arguments)
    
    @property
    def transitivity_pattern(self) -> str:
        """Classify basic transitivity."""
        has_subj = self.has_subject
        has_obj = self.has_object
        has_iobj = any(a.deprel == "iobj" for a in self.arguments)
        
        if has_iobj:
            return "ditransitive"
        elif has_obj:
            return "transitive"
        elif has_subj:
            return "intransitive"
        else:
            return "impersonal"
    
    @property 
    def argument_signature(self) -> str:
        """Create a signature of argument types (for grouping)."""
        deprels = sorted(set(a.deprel.split(":")[0] for a in self.arguments))
        return "+".join(deprels) if deprels else "none"


class VerbExtractor:
    """Extracts verb instances with arguments from parsed sentences."""
    
    # Core argument relations to extract
    CORE_DEPRELS = {
        "nsubj", "nsubj:pass", "nsubj:outer",  # Subjects
        "obj", "iobj",                          # Objects
        "obl", "obl:arg", "obl:agent",          # Obliques (will also match obl:*)
        "xcomp", "ccomp",                       # Clausal complements
        "csubj", "csubj:pass",                  # Clausal subjects
    }
    
    def __init__(self, language: str, use_form_as_lemma: bool = False):
        """
        Initialize extractor.
        
        Args:
            language: Language code
            use_form_as_lemma: If True, use wordform instead of lemma 
                              (for languages without lemmatization like Telugu)
        """
        self.language = language
        self.use_form_as_lemma = use_form_as_lemma
    
    def _get_lemma(self, token: Token) -> str:
        """Get lemma or form depending on configuration."""
        if self.use_form_as_lemma or token.lemma == "_":
            # For Telugu or missing lemmas, use transliteration if available
            translit = token.misc.get("Translit")
            if translit:
                return translit
            return token.form
        return token.lemma
    
    def _is_core_deprel(self, deprel: str) -> bool:
        """Check if deprel is a core argument relation."""
        # Check exact match
        if deprel in self.CORE_DEPRELS:
            return True
        # Check prefix match for subtypes (e.g., obl:tmod, obl:loc)
        base = deprel.split(":")[0]
        return base in {"obl", "nsubj", "csubj"}
    
    def _get_adposition(self, sentence: Sentence, arg_token: Token) -> Optional[str]:
        """Find adposition governing an argument (for obliques)."""
        for token in sentence.get_words():
            if token.head == arg_token.id and token.deprel == "case":
                return self._get_lemma(token)
        return None
    
    def _extract_argument(self, sentence: Sentence, verb: Token, arg_token: Token) -> Argument:
        """Extract argument information from a dependent token."""
        # Get case marking
        case = arg_token.feats.get("Case")
        
        # Get adposition if present
        adposition = self._get_adposition(sentence, arg_token)
        
        # Determine position relative to verb
        head_position = 1 if arg_token.id > verb.id else -1
        
        return Argument(
            deprel=arg_token.deprel,
            form=arg_token.form,
            lemma=self._get_lemma(arg_token),
            upos=arg_token.upos,
            case=case,
            adposition=adposition,
            head_position=head_position,
        )
    
    def extract_from_sentence(self, sentence: Sentence) -> list[VerbInstance]:
        """
        Extract all verb instances from a sentence.
        
        Args:
            sentence: Parsed sentence
            
        Returns:
            List of VerbInstance objects
        """
        instances = []
        words = sentence.get_words()
        
        for token in words:
            # Skip non-verbs
            if token.upos != "VERB":
                continue
            
            # Extract morphological features
            feats = token.feats
            
            # Create verb instance
            instance = VerbInstance(
                lemma=self._get_lemma(token),
                form=token.form,
                verb_id=token.id,
                voice=feats.get("Voice"),
                tense=feats.get("Tense"),
                aspect=feats.get("Aspect"),
                mood=feats.get("Mood"),
                person=feats.get("Person"),
                number=feats.get("Number"),
                sent_id=sentence.sent_id,
                sentence_text=sentence.text,
                language=self.language,
            )
            
            # Extract arguments (direct dependents with core relations)
            for dep in sentence.get_dependents(token.id):
                if self._is_core_deprel(dep.deprel):
                    arg = self._extract_argument(sentence, token, dep)
                    instance.arguments.append(arg)
            
            instances.append(instance)
        
        return instances


@dataclass 
class ExtractionStats:
    """Statistics from verb extraction."""
    language: str
    total_sentences: int
    total_verb_instances: int
    unique_verb_lemmas: int
    
    # Argument statistics
    instances_with_subject: int
    instances_with_object: int
    instances_with_oblique: int
    instances_with_clausal: int
    
    # Transitivity distribution
    transitivity_distribution: dict[str, int]
    
    # Argument relation distribution
    deprel_distribution: dict[str, int]
    
    # Top verb lemmas
    top_verbs: list[tuple[str, int]]
    
    def to_dict(self) -> dict:
        return asdict(self)


def extract_language(
    loader: TreebankLoader, 
    language: str,
    output_path: Path,
    use_form_as_lemma: bool = False,
) -> ExtractionStats:
    """
    Extract all verb instances for a language and write to JSONL.
    
    Args:
        loader: TreebankLoader instance
        language: Language to extract
        output_path: Path for output JSONL file
        use_form_as_lemma: Use wordform instead of lemma
        
    Returns:
        ExtractionStats for the language
    """
    extractor = VerbExtractor(language, use_form_as_lemma=use_form_as_lemma)
    
    # Counters
    total_sentences = 0
    total_instances = 0
    verb_lemma_counts: dict[str, int] = {}
    transitivity_counts: dict[str, int] = {}
    deprel_counts: dict[str, int] = {}
    
    with_subject = 0
    with_object = 0
    with_oblique = 0
    with_clausal = 0
    
    # Process and write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in loader.iter_language(language):
            total_sentences += 1
            
            for instance in extractor.extract_from_sentence(sentence):
                total_instances += 1
                
                # Write instance
                f.write(json.dumps(instance.to_dict(), ensure_ascii=False) + "\n")
                
                # Update counts
                verb_lemma_counts[instance.lemma] = verb_lemma_counts.get(instance.lemma, 0) + 1
                
                trans = instance.transitivity_pattern
                transitivity_counts[trans] = transitivity_counts.get(trans, 0) + 1
                
                for arg in instance.arguments:
                    base_deprel = arg.deprel.split(":")[0]
                    deprel_counts[base_deprel] = deprel_counts.get(base_deprel, 0) + 1
                
                if instance.has_subject:
                    with_subject += 1
                if instance.has_object:
                    with_object += 1
                if instance.has_oblique:
                    with_oblique += 1
                if instance.has_clausal_complement:
                    with_clausal += 1
    
    # Get top verbs
    top_verbs = sorted(verb_lemma_counts.items(), key=lambda x: -x[1])[:20]
    
    return ExtractionStats(
        language=language,
        total_sentences=total_sentences,
        total_verb_instances=total_instances,
        unique_verb_lemmas=len(verb_lemma_counts),
        instances_with_subject=with_subject,
        instances_with_object=with_object,
        instances_with_oblique=with_oblique,
        instances_with_clausal=with_clausal,
        transitivity_distribution=dict(sorted(transitivity_counts.items())),
        deprel_distribution=dict(sorted(deprel_counts.items())),
        top_verbs=top_verbs,
    )


def load_verb_instances(filepath: Path) -> list[VerbInstance]:
    """Load verb instances from JSONL file."""
    instances = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                instances.append(VerbInstance.from_dict(json.loads(line)))
    return instances


def spot_check_samples(filepath: Path, n: int = 10, seed: int = 42) -> list[dict]:
    """
    Load random samples for manual inspection.
    
    Args:
        filepath: Path to JSONL file
        n: Number of samples
        seed: Random seed
        
    Returns:
        List of instance dictionaries
    """
    instances = load_verb_instances(filepath)
    random.seed(seed)
    samples = random.sample(instances, min(n, len(instances)))
    return [inst.to_dict() for inst in samples]


def run_extraction(data_dir: Path, output_dir: Path) -> dict:
    """
    Run extraction for all available languages.
    
    Args:
        data_dir: Path to data/raw/
        output_dir: Path to outputs/
        
    Returns:
        Dictionary of stats per language
    """
    loader = TreebankLoader(data_dir)
    available = loader.available_languages()
    
    print(f"Extracting verbs from {len(available)} language(s)...")
    print()
    
    all_stats = {}
    
    for language in available:
        print(f"Processing {language}...")
        
        # Telugu needs special handling (no lemmas)
        use_form = (language == "telugu")
        
        output_path = output_dir / f"{language}_verb_instances.jsonl"
        stats = extract_language(loader, language, output_path, use_form_as_lemma=use_form)
        all_stats[language] = stats.to_dict()
        
        # Print summary
        print(f"  Verb instances: {stats.total_verb_instances:,}")
        print(f"  Unique lemmas: {stats.unique_verb_lemmas:,}")
        print(f"  With subject: {stats.instances_with_subject:,} ({100*stats.instances_with_subject/stats.total_verb_instances:.1f}%)")
        print(f"  With object: {stats.instances_with_object:,} ({100*stats.instances_with_object/stats.total_verb_instances:.1f}%)")
        print(f"  Transitivity: {stats.transitivity_distribution}")
        print(f"  Output: {output_path}")
        print()
    
    # Write combined stats
    stats_path = output_dir / "extraction_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"Stats written to: {stats_path}")
    
    return all_stats


# CLI interface
if __name__ == "__main__":
    import sys
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    output_dir = project_root / "outputs"
    
    print("=" * 60)
    print("VerbTopology: Verb & Argument Extraction")
    print("=" * 60)
    print()
    
    stats = run_extraction(data_dir, output_dir)
    
    # Spot check
    print()
    print("=" * 60)
    print("Spot Check Samples (English)")
    print("=" * 60)
    
    eng_path = output_dir / "english_verb_instances.jsonl"
    if eng_path.exists():
        samples = spot_check_samples(eng_path, n=5)
        for i, sample in enumerate(samples, 1):
            print(f"\n--- Sample {i} ---")
            print(f"Verb: {sample['lemma']} ({sample['form']})")
            print(f"Sentence: {sample.get('sentence_text', 'N/A')}")
            print(f"Arguments:")
            for arg in sample['arguments']:
                print(f"  - {arg['deprel']}: {arg['lemma']} ({arg['form']})", end="")
                if arg.get('case'):
                    print(f" [Case={arg['case']}]", end="")
                if arg.get('adposition'):
                    print(f" [adp={arg['adposition']}]", end="")
                print()

