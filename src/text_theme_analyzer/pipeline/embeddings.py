"""Disk-backed, content-addressed embedding cache + corpus embedder.

Cache layout (sharded by first 2 hex chars of the hash to keep dir sizes bounded):

    {cache_dir}/embeddings/{model_safe_name}/{sha256[:2]}/{sha256}.npy

Atomic writes: write to `.tmp` first, then rename, so a crash mid-write
can't leave a half-written file that the next run will load as a valid
zero-vector.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from text_theme_analyzer.utils.hashing import normalize_text, sha256_hex
from text_theme_analyzer.utils.progress import progress


def _model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("\\", "__")


@dataclass
class EmbeddingCache:
    root: Path
    model_name: str

    def __post_init__(self) -> None:
        self.model_dir = (
            self.root
            / "embeddings"
            / _model_safe_name(self.model_name)
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def key(self, text: str) -> str:
        return sha256_hex(normalize_text(text))

    def path_for(self, text: str) -> Path:
        k = self.key(text)
        return self.model_dir / k[:2] / f"{k}.npy"

    def get(self, text: str) -> np.ndarray | None:
        path = self.path_for(text)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None

    def put(self, text: str, vec: np.ndarray) -> None:
        path = self.path_for(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use a sibling `.tmp` file for atomic write; np.save adds `.npy` if
        # the path doesn't end in `.npy`, so we use open() + np.save to a
        # file handle so we control the exact path.
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, vec, allow_pickle=False)
        os.replace(tmp, path)

    def cached_count(self, texts: Iterable[str]) -> int:
        return sum(1 for t in texts if self.path_for(t).exists())

    def load_all(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        """Return (vectors, missing_indices) where vectors[i] is the cached vec or None."""
        out: list[np.ndarray | None] = []
        missing: list[int] = []
        for i, t in enumerate(texts):
            v = self.get(t)
            out.append(v)
            if v is None:
                missing.append(i)
        return out, missing


def embed_corpus(
    texts: list[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    cache: EmbeddingCache | None = None,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed a corpus of texts, using cache when present.

    Returns an (N, D) float32 array. Order matches `texts`.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    cache = cache or EmbeddingCache(
        root=Path.home() / ".cache" / "text-theme-analyzer",
        model_name=model_name,
    )
    cached, missing = cache.load_all(texts)
    if not missing:
        return np.stack(cached, axis=0).astype(np.float32)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    missing_texts = [texts[i] for i in missing]
    chunks_iter = list(range(0, len(missing_texts), batch_size))
    if show_progress:
        from tqdm import tqdm
        chunks_iter = tqdm(chunks_iter, desc=f"embedding {len(missing_texts)} new")
    for start in chunks_iter:
        batch = missing_texts[start:start + batch_size]
        vecs = model.encode(batch, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)
        for j, vec in enumerate(vecs):
            global_i = missing[start + j]
            cached[global_i] = vec
            cache.put(texts[global_i], vec.astype(np.float32))
    return np.stack(cached, axis=0).astype(np.float32)
