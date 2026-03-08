"""
ingest.py — Part 1a: Corpus ingestion and preprocessing
=========================================================

Design philosophy
-----------------
The 20 Newsgroups dataset is intentionally messy. Before we can embed anything
meaningfully, we must make deliberate choices about what signal to keep and what
noise to strip. Bad cleaning = polluted embedding space = poor retrieval.

Key noise sources and our treatment of each:
--------------------------------------------

1. NNTP/Email headers (From:, Path:, Xref:, Message-ID:, NNTP-Posting-Host:, etc.)
   Decision: STRIP entirely. Headers reveal sender identity and server routing, not
   topic content. Keeping them would let the model cluster on sender domains rather
   than subject matter — a common mistake that inflates apparent cluster quality.
   Exception: We retain the Subject: line (stripped of "Re:" prefixes) because it
   is a human-authored topical signal, often the most information-dense part of a post.

2. Quoted reply blocks (lines starting with ">", "> >", etc.)
   Decision: STRIP. Quoted text duplicates content from other posts. Retaining it
   would cause posts that heavily quote others to "drift" semantically toward the
   quoted post's topic. This is especially bad for talk.* groups where posts are
   long chains of contentious back-and-forth.

3. Signature blocks ("--" separator at end of posts)
   Decision: STRIP everything after the "-- " separator. Signatures contain personal
   contact info, taglines, and PGP keys — zero topical signal, significant noise.

4. URLs and email addresses
   Decision: STRIP. These are sparse tokens that appear once or twice in the corpus;
   they inflate vocabulary without contributing semantic meaning to embeddings.

5. Encoding artifacts
   Decision: Read all files with latin-1 (the original encoding), then normalize
   to ASCII-safe text. The corpus predates Unicode; forcing UTF-8 silently corrupts
   characters in many posts.

6. Very short documents (< 30 tokens after cleaning)
   Decision: DROP. Posts this short are typically: "I agree", one-line jokes, or
   formatting artifacts. They add noise to cluster centroids and their embeddings
   are unreliable — TF-IDF vectors for 5-word documents have very high variance.

7. Duplicate / near-duplicate posts
   Decision: We flag them via exact SHA-256 hash deduplication. Near-dupes (same
   post cross-posted to multiple groups) are kept only once (first occurrence wins).
   Cross-posting is extremely common in this corpus — not deduplicating inflates
   cluster sizes and biases centroids.

Embedding model choice: TF-IDF + Truncated SVD (LSA)
------------------------------------------------------
We use Latent Semantic Analysis rather than a neural sentence encoder. Reasons:

- No network access to download transformer weights in this environment.
- For keyword-heavy technical newsgroup text, TF-IDF captures topic signal well.
- SVD (LSA) at 256 dimensions removes noise from the raw TF-IDF space and creates
  dense vectors where cosine similarity is meaningful — essential for clustering.
- It's interpretable: SVD components correspond to latent topics.
- Scales to 20k documents in seconds, not minutes.

Trade-off acknowledged: Neural embeddings (e.g., all-MiniLM-L6-v2) would capture
paraphrase similarity better. LSA will miss semantic equivalence across vocabulary
(e.g., "car" vs "automobile"). For a newsgroup corpus with highly consistent
vocabulary per topic, this is an acceptable trade-off.

Vector store choice: In-process NumPy + disk persistence (NPZ + JSON)
----------------------------------------------------------------------
No Chroma, Pinecone, or Weaviate — we build a minimal but complete vector store
from scratch. This makes the architecture transparent and removes dependencies.
The store supports:
- Cosine similarity search with optional category filters
- O(n) brute-force scan, appropriate for 20k vectors at 256d (< 10ms per query)
- Persistence via numpy .npz for vectors + JSON for metadata
- For larger corpora, swap the scan for a FAISS flat index — the interface is
  identical, only the backend changes.
"""

import os
import re
import json
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CORPUS_ROOT = Path(__file__).parent.parent / "20_newsgroups"                  # dataset folder, relative to CWD
OUT_DIR     = Path(__file__).parent.parent           # project root
DATA_DIR    = OUT_DIR / "data"
EMB_DIR     = OUT_DIR / "embeddings"
VS_DIR      = OUT_DIR / "vector_store"

for d in [DATA_DIR, EMB_DIR, VS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding hyper-parameters — justified in docstring above
# ---------------------------------------------------------------------------
SVD_DIMS    = 256   # Dimensionality of final embedding vectors
TFIDF_MAX_F = 50000 # Vocabulary cap; keeps memory bounded and removes ultra-rare tokens
TFIDF_MIN_DF = 3    # Token must appear in at least 3 docs (prunes hapax legomena)
TFIDF_MAX_DF = 0.85 # Token in >85% of docs is a stopword-equivalent (e.g. "writes")
MIN_TOKENS  = 30    # Drop docs shorter than this after cleaning


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
# Pre-compiled for performance — applied to every post
_HEADER_RE    = re.compile(r'^[A-Za-z\-]+:.*$', re.MULTILINE)
_QUOTE_RE     = re.compile(r'^>.*$', re.MULTILINE)
_URL_RE       = re.compile(r'https?://\S+|www\.\S+')
_EMAIL_RE     = re.compile(r'\S+@\S+\.\S+')
_SIG_RE       = re.compile(r'\n--\s*\n.*', re.DOTALL)
_WHITESPACE_RE = re.compile(r'\s+')


def _extract_subject(raw: str) -> str:
    """Pull the Subject: header value, strip Re:/Fwd: prefixes."""
    m = re.search(r'^Subject:\s*(.+)$', raw, re.MULTILINE | re.IGNORECASE)
    if not m:
        return ""
    subj = m.group(1).strip()
    # Strip "Re: Re: Re:" chains — only the base topic matters
    subj = re.sub(r'^(Re|Fwd|Fw):\s*', '', subj, flags=re.IGNORECASE).strip()
    return subj


def clean_post(raw: str) -> str:
    """
    Strip all noise from a raw newsgroup post, returning clean body text.
    The subject line is prepended once (as a topical anchor) then the rest
    of the header block is removed entirely.
    """
    # 1. Capture subject before we nuke all headers
    subject = _extract_subject(raw)

    # 2. Find the blank line that separates headers from body
    #    NNTP format guarantees headers end at first blank line
    parts = re.split(r'\n\n', raw, maxsplit=1)
    body = parts[1] if len(parts) > 1 else raw

    # 3. Strip signature block (everything after "-- " on its own line)
    body = _SIG_RE.sub('', body)

    # 4. Strip quoted reply lines
    body = _QUOTE_RE.sub('', body)

    # 5. Strip URLs and email addresses
    body = _URL_RE.sub(' ', body)
    body = _EMAIL_RE.sub(' ', body)

    # 6. Strip residual header-like lines that leak into the body (common
    #    when posts are forwarded or have inline headers)
    body = _HEADER_RE.sub(' ', body)

    # 7. Collapse whitespace
    body = _WHITESPACE_RE.sub(' ', body).strip()

    # 8. Prepend subject as topical anchor (weighted by appearing at front)
    if subject:
        return f"{subject} {subject} {body}"  # doubled for mild TF boost
    return body


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(corpus_root: Path = CORPUS_ROOT) -> list[dict]:
    """
    Load all posts, clean them, deduplicate, and return a list of records.

    Each record:
        {
            "id":       str,   # SHA-256[:16] of cleaned text
            "doc_id":   str,   # original filename
            "category": str,   # newsgroup name
            "text":     str,   # cleaned body text
            "n_tokens": int,   # approximate whitespace-token count
        }
    """
    records = []
    seen_hashes: set[str] = set()
    dropped_short = 0
    dropped_dupe  = 0
    total_raw     = 0

    categories = sorted([d.name for d in corpus_root.iterdir() if d.is_dir()])
    log.info(f"Found {len(categories)} categories in corpus root")

    for cat in categories:
        cat_dir = corpus_root / cat
        files = sorted(cat_dir.iterdir())

        for fpath in files:
            if not fpath.is_file():
                continue
            total_raw += 1

            # Read with latin-1: the corpus was written before Unicode.
            # Forcing UTF-8 silently corrupts ~5% of posts.
            try:
                raw = fpath.read_text(encoding='latin-1', errors='replace')
            except Exception as e:
                log.warning(f"Could not read {fpath}: {e}")
                continue

            cleaned = clean_post(raw)
            n_tokens = len(cleaned.split())

            # Drop very short posts — their embeddings are high-variance noise
            if n_tokens < MIN_TOKENS:
                dropped_short += 1
                continue

            # Exact-duplicate detection via hash of cleaned text.
            # Cross-posting is extremely common; same post appears in 2-4 groups.
            # We keep first occurrence (preserves the "home" category label).
            doc_hash = hashlib.sha256(cleaned.encode()).hexdigest()[:16]
            if doc_hash in seen_hashes:
                dropped_dupe += 1
                continue
            seen_hashes.add(doc_hash)

            records.append({
                "id":       doc_hash,
                "doc_id":   fpath.name,
                "category": cat,
                "text":     cleaned,
                "n_tokens": n_tokens,
            })

    log.info(
        f"Loaded {len(records)} documents from {total_raw} raw posts | "
        f"dropped {dropped_short} (too short) + {dropped_dupe} (duplicates)"
    )
    return records


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------

def build_embeddings(records: list[dict]) -> tuple[np.ndarray, TfidfVectorizer, TruncatedSVD]:
    """
    Build L2-normalised LSA embeddings for all documents.

    Pipeline: raw text → TF-IDF sparse matrix → SVD → L2 normalisation

    Why L2 normalise?
    All downstream operations (cosine similarity, clustering, cache lookup)
    reduce to dot products on unit vectors after normalisation. This is faster
    and numerically cleaner than computing cosine distances explicitly.
    """
    texts = [r["text"] for r in records]

    log.info("Fitting TF-IDF vectorizer…")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_F,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,    # log(1+tf) dampens frequency extremes; improves LSA
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b',  # alpha-start, len ≥ 3
        ngram_range=(1, 2),   # unigrams + bigrams capture phrases like "space shuttle"
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    log.info(f"TF-IDF matrix: {tfidf_matrix.shape}  (docs × vocab)")

    log.info(f"Running Truncated SVD to {SVD_DIMS} dimensions…")
    svd = TruncatedSVD(n_components=SVD_DIMS, random_state=42, n_iter=5)
    embeddings = svd.fit_transform(tfidf_matrix)

    # L2 normalise so cosine similarity = dot product
    embeddings = normalize(embeddings, norm='l2')

    explained = svd.explained_variance_ratio_.sum()
    log.info(
        f"SVD complete. Embeddings: {embeddings.shape}. "
        f"Explained variance: {explained:.3f} ({explained*100:.1f}%)"
    )

    return embeddings, vectorizer, svd


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(records, embeddings, vectorizer, svd):
    """Persist all artifacts to disk for use by downstream modules."""

    # 1. Cleaned corpus as JSONL
    corpus_path = DATA_DIR / "corpus.jsonl"
    with open(corpus_path, 'w') as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != 'text'}) + '\n')
    # Save texts separately (large)
    texts_path = DATA_DIR / "texts.json"
    with open(texts_path, 'w') as f:
        json.dump({r["id"]: r["text"] for r in records}, f)
    log.info(f"Corpus saved → {corpus_path}")

    # 2. Embeddings as compressed numpy array
    emb_path = EMB_DIR / "embeddings.npz"
    ids = np.array([r["id"] for r in records])
    np.savez_compressed(emb_path, embeddings=embeddings, ids=ids)
    log.info(f"Embeddings saved → {emb_path}  {embeddings.shape}")

    # 3. Sklearn pipeline objects (vectorizer + SVD)
    with open(EMB_DIR / "vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(EMB_DIR / "svd.pkl", 'wb') as f:
        pickle.dump(svd, f)
    log.info("Vectorizer and SVD saved.")

    # 4. Category → index mapping for filtered search
    cat_index: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        cat_index.setdefault(r["category"], []).append(i)
    with open(VS_DIR / "category_index.json", 'w') as f:
        json.dump(cat_index, f)

    # 5. Full metadata for vector store
    meta = [{"id": r["id"], "doc_id": r["doc_id"],
              "category": r["category"], "n_tokens": r["n_tokens"]}
            for r in records]
    with open(VS_DIR / "metadata.json", 'w') as f:
        json.dump(meta, f)
    log.info("Vector store metadata saved.")

    # 6. Summary stats
    stats = {
        "n_docs": len(records),
        "n_categories": len(set(r["category"] for r in records)),
        "svd_dims": int(SVD_DIMS),
        "vocab_size": len(vectorizer.vocabulary_),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "categories": sorted(set(r["category"] for r in records)),
    }
    with open(OUT_DIR / "corpus_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    log.info(f"Stats: {stats}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("Part 1a — Ingesting and embedding the 20 Newsgroups corpus")
    log.info("=" * 60)

    records    = load_corpus()
    embeddings, vectorizer, svd = build_embeddings(records)
    save_artifacts(records, embeddings, vectorizer, svd)

    log.info("Ingestion complete. All artifacts written to disk.")
