"""
Data loader for CoNLL-U treebank files.

Provides unified parsing interface for all languages in the VerbTopology project.
Handles multi-word tokens (MWTs) and extracts sentence/token-level data.
"""

from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field
import json

import conllu
from conllu import TokenList


@dataclass
class Token:
    """Represents a single token from a CoNLL-U file."""
    id: int | tuple  # int for regular tokens, tuple for MWTs (e.g., (1, 2))
    form: str
    lemma: str
    upos: str
    xpos: Optional[str]
    feats: dict = field(default_factory=dict)
    head: Optional[int] = None
    deprel: Optional[str] = None
    deps: Optional[str] = None
    misc: dict = field(default_factory=dict)
    
    @classmethod
    def from_conllu(cls, token: dict) -> "Token":
        """Create Token from conllu library token dict."""
        return cls(
            id=token["id"],
            form=token["form"],
            lemma=token.get("lemma", "_"),
            upos=token.get("upos", "_"),
            xpos=token.get("xpos"),
            feats=token.get("feats") or {},
            head=token.get("head"),
            deprel=token.get("deprel"),
            deps=token.get("deps"),
            misc=token.get("misc") or {},
        )
    
    def is_multiword(self) -> bool:
        """Check if this is a multi-word token range."""
        return isinstance(self.id, tuple)
    
    def is_empty_node(self) -> bool:
        """Check if this is an empty node (decimal ID like 2.1)."""
        return isinstance(self.id, float)


@dataclass
class Sentence:
    """Represents a parsed sentence with metadata and tokens."""
    sent_id: Optional[str]
    text: Optional[str]
    tokens: list[Token]
    metadata: dict = field(default_factory=dict)
    
    @classmethod
    def from_token_list(cls, token_list: TokenList) -> "Sentence":
        """Create Sentence from conllu TokenList."""
        tokens = [Token.from_conllu(t) for t in token_list]
        metadata = dict(token_list.metadata) if token_list.metadata else {}
        
        return cls(
            sent_id=metadata.get("sent_id"),
            text=metadata.get("text"),
            tokens=tokens,
            metadata=metadata,
        )
    
    def get_words(self) -> list[Token]:
        """Get only regular word tokens (excluding MWTs and empty nodes)."""
        return [t for t in self.tokens 
                if isinstance(t.id, int)]
    
    def get_by_id(self, token_id: int) -> Optional[Token]:
        """Get token by its ID."""
        for t in self.tokens:
            if t.id == token_id:
                return t
        return None
    
    def get_dependents(self, head_id: int) -> list[Token]:
        """Get all tokens that depend on the given head."""
        return [t for t in self.get_words() if t.head == head_id]


class TreebankLoader:
    """Loads and parses CoNLL-U treebank files."""
    
    def __init__(self, data_dir: Path | str):
        """
        Initialize loader with data directory.
        
        Args:
            data_dir: Path to data/raw/ directory containing language subdirs
        """
        self.data_dir = Path(data_dir)
        
        # Expected language directories
        self.language_dirs = {
            "english": self.data_dir / "english",
            "telugu": self.data_dir / "telugu", 
            "vedic": self.data_dir / "vedic",
            "sanskrit": self.data_dir / "sanskrit",
        }
    
    def get_conllu_files(self, language: str) -> list[Path]:
        """Get all CoNLL-U files for a language."""
        lang_dir = self.language_dirs.get(language)
        if lang_dir is None:
            raise ValueError(f"Unknown language: {language}")
        
        if not lang_dir.exists():
            raise FileNotFoundError(f"Language directory not found: {lang_dir}")
        
        files = list(lang_dir.glob("*.conllu"))
        if not files:
            raise FileNotFoundError(f"No .conllu files found in {lang_dir}")
        
        return sorted(files)
    
    def parse_file(self, filepath: Path) -> Iterator[Sentence]:
        """
        Parse a single CoNLL-U file.
        
        Args:
            filepath: Path to .conllu file
            
        Yields:
            Sentence objects
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read()
        
        for token_list in conllu.parse(data):
            yield Sentence.from_token_list(token_list)
    
    def load_language(self, language: str) -> list[Sentence]:
        """
        Load all sentences for a language.
        
        Args:
            language: One of 'english', 'telugu', 'vedic', 'sanskrit'
            
        Returns:
            List of all Sentence objects
        """
        sentences = []
        for filepath in self.get_conllu_files(language):
            sentences.extend(self.parse_file(filepath))
        return sentences
    
    def iter_language(self, language: str) -> Iterator[Sentence]:
        """
        Iterate over sentences for a language (memory-efficient).
        
        Args:
            language: One of 'english', 'telugu', 'vedic', 'sanskrit'
            
        Yields:
            Sentence objects
        """
        for filepath in self.get_conllu_files(language):
            yield from self.parse_file(filepath)
    
    def available_languages(self) -> list[str]:
        """Get list of languages with data available."""
        available = []
        for lang, lang_dir in self.language_dirs.items():
            if lang_dir.exists():
                files = list(lang_dir.glob("*.conllu"))
                if files:
                    available.append(lang)
        return available


@dataclass
class TreebankStats:
    """Statistics for a treebank."""
    language: str
    num_files: int
    num_sentences: int
    num_tokens: int
    num_words: int  # excluding MWTs
    unique_lemmas: int
    unique_forms: int
    upos_distribution: dict[str, int]
    verb_count: int
    verb_lemmas: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "language": self.language,
            "num_files": self.num_files,
            "num_sentences": self.num_sentences,
            "num_tokens": self.num_tokens,
            "num_words": self.num_words,
            "unique_lemmas": self.unique_lemmas,
            "unique_forms": self.unique_forms,
            "upos_distribution": self.upos_distribution,
            "verb_count": self.verb_count,
            "verb_lemmas": self.verb_lemmas,
        }


def compute_treebank_stats(loader: TreebankLoader, language: str) -> TreebankStats:
    """
    Compute statistics for a treebank.
    
    Args:
        loader: TreebankLoader instance
        language: Language to compute stats for
        
    Returns:
        TreebankStats object
    """
    num_files = len(loader.get_conllu_files(language))
    num_sentences = 0
    num_tokens = 0
    num_words = 0
    
    lemmas = set()
    forms = set()
    upos_counts: dict[str, int] = {}
    verb_lemmas = set()
    verb_count = 0
    
    for sentence in loader.iter_language(language):
        num_sentences += 1
        num_tokens += len(sentence.tokens)
        
        for token in sentence.get_words():
            num_words += 1
            lemmas.add(token.lemma)
            forms.add(token.form)
            
            upos = token.upos
            upos_counts[upos] = upos_counts.get(upos, 0) + 1
            
            if upos == "VERB":
                verb_count += 1
                verb_lemmas.add(token.lemma)
    
    return TreebankStats(
        language=language,
        num_files=num_files,
        num_sentences=num_sentences,
        num_tokens=num_tokens,
        num_words=num_words,
        unique_lemmas=len(lemmas),
        unique_forms=len(forms),
        upos_distribution=dict(sorted(upos_counts.items())),
        verb_count=verb_count,
        verb_lemmas=len(verb_lemmas),
    )


def verify_data(data_dir: Path | str, output_path: Optional[Path | str] = None) -> dict:
    """
    Verify all treebank data and generate summary statistics.
    
    Args:
        data_dir: Path to data/raw/ directory
        output_path: Optional path to write JSON summary
        
    Returns:
        Dictionary with stats for all available languages
    """
    loader = TreebankLoader(data_dir)
    available = loader.available_languages()
    
    if not available:
        print("WARNING: No languages with data found!")
        print(f"Expected data in: {loader.data_dir}")
        print("Please add .conllu files to language subdirectories:")
        for lang, path in loader.language_dirs.items():
            print(f"  - {lang}: {path}")
        return {}
    
    print(f"Found data for {len(available)} language(s): {', '.join(available)}")
    print()
    
    results = {}
    for language in available:
        print(f"Processing {language}...")
        try:
            stats = compute_treebank_stats(loader, language)
            results[language] = stats.to_dict()
            
            print(f"  Files: {stats.num_files}")
            print(f"  Sentences: {stats.num_sentences:,}")
            print(f"  Words: {stats.num_words:,}")
            print(f"  Unique lemmas: {stats.unique_lemmas:,}")
            print(f"  Verbs: {stats.verb_count:,} ({stats.verb_lemmas:,} lemmas)")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[language] = {"error": str(e)}
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Summary written to: {output_path}")
    
    return results


# CLI interface
if __name__ == "__main__":
    import sys
    
    # Default paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "raw"
    output_path = project_root / "outputs" / "data_summary.json"
    
    # Allow override via command line
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    
    print("=" * 60)
    print("VerbTopology: Data Verification")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print()
    
    verify_data(data_dir, output_path)

