"""Task similarity providers for kernel-weighted evidence aggregation.

The HSTG belief backend weights historical trajectory evidence by the
semantic similarity between the current task and each recorded task. The
default provider is lexical and depends only on the Python standard
library, which preserves the core package's stdlib-only runtime. An
embedding-backed provider can be plugged in for experiments that have an
embedding endpoint available.
"""

from __future__ import annotations

import math
import re
from typing import Callable, Dict, List, Sequence

try:  # Python 3.8+
    from typing import Protocol
except ImportError:  # pragma: no cover
    Protocol = object


_TOKEN_RE = re.compile(r"[a-z0-9_.]+|[一-鿿]")


def tokenize(text: str) -> List[str]:
    """Split text into lowercase latin/numeric tokens and single CJK characters."""

    return _TOKEN_RE.findall(str(text or "").lower())


class SimilarityProvider(Protocol):
    """Return a similarity kernel value K(a, b) in [0, 1]."""

    def similarity(self, a: str, b: str) -> float:  # pragma: no cover - protocol
        ...


class LexicalSimilarity:
    """Stdlib-only lexical kernel: unigram and bigram Jaccard overlap."""

    def __init__(self, unigram_weight: float = 0.6, bigram_weight: float = 0.4):
        total = float(unigram_weight) + float(bigram_weight)
        self.unigram_weight = float(unigram_weight) / total if total else 0.5
        self.bigram_weight = float(bigram_weight) / total if total else 0.5

    def similarity(self, a: str, b: str) -> float:
        tokens_a = tokenize(a)
        tokens_b = tokenize(b)
        if not tokens_a or not tokens_b:
            return 0.0
        score = self.unigram_weight * _jaccard(set(tokens_a), set(tokens_b))
        score += self.bigram_weight * _jaccard(_bigrams(tokens_a), _bigrams(tokens_b))
        return max(0.0, min(1.0, score))


class EmbeddingSimilarity:
    """Optional kernel backed by an injected embedding function.

    ``embed`` maps a batch of texts to vectors. Vectors are cached per text
    so repeated kernel evaluations against the same evidence set stay cheap.
    Cosine similarity is clamped to [0, 1] to keep the kernel valid.
    """

    def __init__(self, embed: Callable[[Sequence[str]], Sequence[Sequence[float]]]):
        self.embed = embed
        self._cache: Dict[str, List[float]] = {}

    def similarity(self, a: str, b: str) -> float:
        vec_a = self._vector(str(a or ""))
        vec_b = self._vector(str(b or ""))
        if not vec_a or not vec_b:
            return 0.0
        return max(0.0, min(1.0, _cosine(vec_a, vec_b)))

    def _vector(self, text: str) -> List[float]:
        if not text:
            return []
        if text not in self._cache:
            self._cache[text] = [float(value) for value in self.embed([text])[0]]
        return self._cache[text]


_DEFAULT_PROVIDER = LexicalSimilarity()


def default_similarity() -> SimilarityProvider:
    return _DEFAULT_PROVIDER


def _jaccard(a, b) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _bigrams(tokens: List[str]):
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)
