"""Task similarity providers for kernel-weighted evidence aggregation.

The HSTG belief backend weights historical trajectory evidence by the
semantic similarity between the current task and each recorded task. The
default provider is lexical and depends only on the Python standard
library, which preserves the core package's stdlib-only runtime. An
embedding-backed provider can be plugged in for experiments that have an
embedding endpoint available.
"""

from __future__ import annotations

import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
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

    def warm(self, texts: Sequence[str]) -> int:
        """Embed uncached texts in one batch call; returns how many were fetched."""

        missing = [str(text) for text in dict.fromkeys(texts) if str(text or "") and str(text) not in self._cache]
        if not missing:
            return 0
        for text, vector in zip(missing, self.embed(missing)):
            self._cache[text] = [float(value) for value in vector]
        return len(missing)

    def _vector(self, text: str) -> List[float]:
        if not text:
            return []
        if text not in self._cache:
            self._cache[text] = [float(value) for value in self.embed([text])[0]]
        return self._cache[text]


class OpenAICompatibleEmbeddingClient:
    """Batch embedding function backed by an OpenAI-compatible /embeddings API.

    Uses only the standard library so the core package stays dependency
    free. Instances are callables suitable for ``EmbeddingSimilarity``:

        provider = EmbeddingSimilarity(OpenAICompatibleEmbeddingClient(
            model="text-embedding-v4",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key_env="EMBEDDING_API_KEY",
        ))
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str = "EMBEDDING_API_KEY",
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        verify_ssl: bool = True,
    ):
        self.model = str(model)
        self.base_url = str(base_url).rstrip("/")
        self.api_key_env = str(api_key_env)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = max(1, int(max_retries))
        self.verify_ssl = bool(verify_ssl)
        if not self.model:
            raise ValueError("Embedding client needs a model name.")
        if not self.base_url:
            raise ValueError("Embedding client needs a base URL, e.g. https://api.openai.com/v1")

    def __call__(self, texts: Sequence[str]) -> List[List[float]]:
        payload = {"model": self.model, "input": [str(text) for text in texts]}
        raw = self._post_json(f"{self.base_url}/embeddings", payload)
        items = sorted(raw.get("data") or [], key=lambda item: int(item.get("index", 0)))
        vectors = [[float(value) for value in item.get("embedding") or []] for item in items]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding API returned {len(vectors)} vectors for {len(texts)} inputs (model={self.model})."
            )
        return vectors

    def _post_json(self, url: str, payload: Dict) -> Dict:
        api_key = os.environ.get(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Embedding API key env var {self.api_key_env} is not set.")
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        context = None if self.verify_ssl else ssl._create_unverified_context()
        last_error: Exception = RuntimeError("embedding request not attempted")
        for attempt in range(self.max_retries):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds, context=context) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"Embedding API request failed after {self.max_retries} attempts: {last_error}") from last_error


_DEFAULT_PROVIDER: SimilarityProvider = LexicalSimilarity()


def default_similarity() -> SimilarityProvider:
    return _DEFAULT_PROVIDER


def set_default_similarity(provider: SimilarityProvider) -> SimilarityProvider:
    """Swap the process-wide default kernel; returns the previous provider.

    HSTG belief states resolve the default provider at call time, so
    setting an embedding-backed provider here switches every kernel
    evaluation (prediction, patch gating, audits) without threading the
    provider through each call site.
    """

    global _DEFAULT_PROVIDER
    previous = _DEFAULT_PROVIDER
    _DEFAULT_PROVIDER = provider
    return previous


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
