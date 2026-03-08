"""
vector_store.py — Part 1b: Vector database with filtered retrieval
===================================================================

Design: A minimal but complete vector store built from first principles.

Architecture decisions
----------------------

Storage layout
~~~~~~~~~~~~~~
- Vectors: NumPy float32 array, shape (N, D), memory-mapped from .npz
  Float32 halves memory vs float64 with negligible precision loss for cosine search.
- Metadata: Plain Python list of dicts, loaded into RAM.
- Category index: dict[category → sorted list of row indices] for O(1) filter setup.

Search algorithm
~~~~~~~~~~~~~~~~
Brute-force cosine similarity (matrix multiply on L2-normalised vectors).
At N=20k, D=256: one query = 20k × 256 float32 dot products = ~5ms on CPU.
This is fast enough for interactive search. We avoid approximate search (HNSW,
IVF-Flat) because:
  - At this scale, exact search is faster than the index overhead.
  - The cache layer (Part 2) means most repeated queries never hit the store.
  - Approximate methods introduce recall trade-offs that are hard to tune blindly.

If the corpus grew to >500k documents, swapping in a FAISS flat index would
require changing only the `_search_vectors` method — the rest of the interface
is unchanged.

Filtering
~~~~~~~~~
Category filter is applied *before* scoring by restricting the row set.
This is faster than post-hoc filtering and avoids retrieving irrelevant docs.

Query embedding
~~~~~~~~~~~~~~~
Query text → same TF-IDF + SVD pipeline used for corpus documents → L2 normalise.
Critically, we use the *fitted* vectorizer and SVD from training — never refit.
Refitting would give different basis vectors, making query vectors incommensurable
with stored document vectors.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import normalize

log = logging.getLogger(__name__)

OUT_DIR  = Path(__file__).parent.parent  # project root
EMB_DIR  = OUT_DIR / "embeddings"
VS_DIR   = OUT_DIR / "vector_store"
DATA_DIR = OUT_DIR / "data"


class VectorStore:
    """
    An in-process vector store with cosine similarity search.

    Usage:
        store = VectorStore.load()
        results = store.search("shuttle launch NASA", top_k=5)
        results = store.search("encryption keys", top_k=5, category="sci.crypt")
    """

    def __init__(
        self,
        embeddings: np.ndarray,       # shape (N, D), L2-normalised float32
        ids: list[str],                # doc id for each row
        metadata: list[dict],          # parallel list of metadata dicts
        category_index: dict,          # category → list of row indices
        vectorizer,                    # fitted TfidfVectorizer
        svd,                           # fitted TruncatedSVD
        texts: dict[str, str],         # id → cleaned text
    ):
        self.embeddings     = embeddings.astype(np.float32)
        self.ids            = ids
        self.metadata       = metadata
        self.category_index = category_index
        self.vectorizer     = vectorizer
        self.svd            = svd
        self.texts          = texts
        self._n, self._d    = embeddings.shape
        log.info(f"VectorStore loaded: {self._n} docs, {self._d}d embeddings")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(cls) -> "VectorStore":
        """Load all artifacts from disk and construct the store."""
        log.info("Loading vector store from disk…")

        npz = np.load(EMB_DIR / "embeddings.npz", allow_pickle=False)
        embeddings = npz["embeddings"]
        ids = npz["ids"].tolist()

        with open(EMB_DIR / "vectorizer.pkl", 'rb') as f:
            vectorizer = pickle.load(f)
        with open(EMB_DIR / "svd.pkl", 'rb') as f:
            svd = pickle.load(f)

        with open(VS_DIR / "metadata.json") as f:
            metadata = json.load(f)
        with open(VS_DIR / "category_index.json") as f:
            category_index = json.load(f)
        with open(DATA_DIR / "texts.json") as f:
            texts = json.load(f)

        return cls(embeddings, ids, metadata, category_index, vectorizer, svd, texts)

    # ------------------------------------------------------------------
    # Query embedding
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a free-text query into the same vector space as the corpus.

        The same TF-IDF → SVD pipeline is applied. We must use transform(),
        never fit_transform(), to stay in the same coordinate system.
        Returns: shape (D,) L2-normalised float32 vector.
        """
        tfidf_vec  = self.vectorizer.transform([text])          # (1, vocab)
        svd_vec    = self.svd.transform(tfidf_vec)               # (1, D)
        normed_vec = normalize(svd_vec, norm='l2')[0]            # (D,)
        return normed_vec.astype(np.float32)

    # ------------------------------------------------------------------
    # Core search
    # ------------------------------------------------------------------

    def _search_vectors(
        self,
        query_vec: np.ndarray,
        row_mask: Optional[np.ndarray] = None,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Brute-force cosine similarity search.

        Because vectors are L2-normalised, cosine_sim(a, b) = a · b.
        Matrix multiply gives scores for all docs in one BLAS call.

        Args:
            query_vec: (D,) query embedding
            row_mask:  optional boolean array of shape (N,) to restrict search
            top_k:     number of results to return

        Returns:
            List of (row_index, score) sorted descending by score.
        """
        if row_mask is not None:
            # Restrict to masked rows only — avoids scoring irrelevant docs
            candidate_indices = np.where(row_mask)[0]
            candidate_vecs    = self.embeddings[candidate_indices]  # (M, D)
            scores            = candidate_vecs @ query_vec          # (M,)
            # Top-k within candidates
            k = min(top_k, len(scores))
            top_local = np.argpartition(scores, -k)[-k:]
            top_local = top_local[np.argsort(scores[top_local])[::-1]]
            return [(int(candidate_indices[i]), float(scores[i])) for i in top_local]
        else:
            scores  = self.embeddings @ query_vec                   # (N,)
            k       = min(top_k, len(scores))
            top_idx = np.argpartition(scores, -k)[-k:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            return [(int(i), float(scores[i])) for i in top_idx]

    def search(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
        return_text: bool = False,
    ) -> list[dict]:
        """
        Search the corpus for documents semantically similar to `query`.

        Args:
            query:       Natural-language search string
            top_k:       Number of results to return
            category:    If set, restrict search to this newsgroup category
            return_text: If True, include the cleaned document text in results

        Returns:
            List of result dicts:
            {
                "rank":     int,
                "score":    float,   # cosine similarity [0, 1]
                "id":       str,
                "category": str,
                "doc_id":   str,
                "n_tokens": int,
                "text":     str,     # only if return_text=True
            }
        """
        query_vec = self.embed_query(query)

        # Build row mask for category filtering
        row_mask = None
        if category:
            if category not in self.category_index:
                raise ValueError(
                    f"Unknown category '{category}'. "
                    f"Valid: {sorted(self.category_index.keys())}"
                )
            indices  = self.category_index[category]
            row_mask = np.zeros(self._n, dtype=bool)
            row_mask[indices] = True

        hits = self._search_vectors(query_vec, row_mask=row_mask, top_k=top_k)

        results = []
        for rank, (row_idx, score) in enumerate(hits, 1):
            meta = self.metadata[row_idx]
            result = {
                "rank":     rank,
                "score":    round(score, 6),
                "id":       meta["id"],
                "category": meta["category"],
                "doc_id":   meta["doc_id"],
                "n_tokens": meta["n_tokens"],
            }
            if return_text:
                result["text"] = self.texts.get(meta["id"], "")
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def categories(self) -> list[str]:
        return sorted(self.category_index.keys())

    @property
    def n_docs(self) -> int:
        return self._n

    def get_doc_by_id(self, doc_id: str) -> Optional[dict]:
        """Retrieve full metadata + text for a document by its hash id."""
        for meta in self.metadata:
            if meta["id"] == doc_id:
                return {**meta, "text": self.texts.get(doc_id, "")}
        return None

    def add_document(self, text: str, category: str, doc_id: str = "custom") -> dict:
        """
        Add a single new document to the store at runtime.
        Useful for testing cache with novel documents.
        """
        import hashlib
        vec     = self.embed_query(text)
        new_id  = hashlib.sha256(text.encode()).hexdigest()[:16]
        n_toks  = len(text.split())

        self.embeddings = np.vstack([self.embeddings, vec[np.newaxis, :]])
        self.ids.append(new_id)
        new_meta = {"id": new_id, "doc_id": doc_id, "category": category, "n_tokens": n_toks}
        self.metadata.append(new_meta)
        self.texts[new_id] = text
        self.category_index.setdefault(category, []).append(self._n)
        self._n += 1
        return new_meta
