# 20 Newsgroups Semantic Search

Semantic search system built on the 20 Newsgroups dataset (~18k posts). Covers embedding, fuzzy clustering, a semantic cache, and a FastAPI service.

---

## Setup

```bash
git clone <repo>
cd <repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset from https://archive.ics.uci.edu/dataset/113/twenty+newsgroups and extract the `20_newsgroups` folder into the project root.

---

## Running the pipeline

**Step 1 — Embed the corpus**

```bash
python data/ingest.py
```

Cleans ~20k posts, builds TF-IDF + SVD(256) embeddings, saves everything to `embeddings/` and `vector_store/`. Takes about 45 seconds.

**Step 2 — Cluster**

```bash
python clustering/fuzzy_cluster.py
```

Runs K selection (Silhouette, Calinski-Harabasz, Gap Statistic) and fits a von Mises-Fisher mixture at K=17. Saves membership matrix and centroids to `clustering/`.

**Step 3 — Start the API**

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
docker-compose up --build
```

---

## API

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "encryption public key RSA"}'
```

```json
{
  "query": "encryption public key RSA",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.0,
  "result": [
    {"rank": 1, "score": 0.779, "category": "sci.crypt", "doc_id": "14989"},
    ...
  ],
  "dominant_cluster": 14
}
```

Run the same (or similar) query again and you'll get a cache hit:

```json
{
  "cache_hit": true,
  "matched_query": "encryption public key RSA",
  "similarity_score": 0.9998,
  ...
}
```

### GET /cache/stats

```bash
curl http://localhost:8000/cache/stats
```

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### DELETE /cache

```bash
curl -X DELETE http://localhost:8000/cache
```

Flushes everything and resets stats.

---

## How it works

### Part 1 — Embeddings

Posts are cleaned before embedding: headers stripped (routing ≠ topic signal), quoted reply blocks removed (they cause topic drift), signatures cut, URLs/emails removed. Short posts and exact duplicates dropped.

Embedding model: TF-IDF over 50k vocab → TruncatedSVD to 256 dimensions (LSA). Fast, works offline, and captures per-topic vocabulary well enough for this corpus.

### Part 2 — Fuzzy clustering

Standard Fuzzy C-Means doesn't work on unit-normalised vectors — in 256D space all Euclidean distance ratios converge to 1, so every document gets identical membership in every cluster. The correct model for data on the unit hypersphere is von Mises-Fisher, where soft assignments are proportional to `exp(κ · cosine_similarity)`. That's what's implemented here.

K was chosen at 17 (not 20) because the embedding space has genuine semantic merges that the editorial category split ignores: hockey and baseball share enough vocabulary to form one cluster, the three `talk.politics.*` groups split into two (domestic vs foreign), and `alt.atheism` + `talk.religion.misc` are the same debate from opposite sides.

The most interesting documents are the ones with high entropy across their membership distribution — a post about gun buyback programs that uses legal language pulls toward politics clusters rather than the firearms cluster. A creationism debate post in `alt.atheism` spreads across religion, science, and politics.

### Part 3 — Semantic cache

The cache stores (query embedding, result) pairs. On each new query, it computes the embedding, checks which cluster it belongs to, and only scans entries in that cluster's bucket rather than the whole cache. This gives roughly an 8× speedup over a flat cache at any size.

The similarity threshold θ=0.85 was chosen by measuring precision and recall across a set of query pairs with known ground truth. The key finding: in TF-IDF/LSA space, the cache catches token-order variations and closely related phrasings, but not synonym substitutions ("gun control" vs "firearms regulations" scores 0.55). That's a model limitation, not a cache limitation — the threshold can't fix it.

### Part 4 — API

Built with FastAPI. State (vector store + cache) is loaded once at startup via the ASGI lifespan protocol and stored on `app.state`. Returns 503 until ready.

---

## Project structure

```
.
├── data/
│   └── ingest.py           # corpus cleaning + TF-IDF/LSA embedding
├── embeddings/             # generated: embeddings.npz, vectorizer.pkl, svd.pkl
├── vector_store/
│   ├── vector_store.py     # cosine similarity search with category filtering
│   └── metadata.json       # generated: doc metadata
├── clustering/
│   ├── fuzzy_cluster.py    # vMF-EM clustering with K selection
│   ├── centroids.npy       # generated
│   └── memberships.npy     # generated: (N, 17) soft membership matrix
├── cache/
│   └── semantic_cache.py   # cluster-aware semantic cache
├── api/
│   └── app.py              # FastAPI endpoints
├── asgi_server.py          # stdlib ASGI server (fallback if uvicorn unavailable)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
