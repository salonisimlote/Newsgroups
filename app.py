"""
api/app.py  —  Part 4: FastAPI-compatible service
==================================================

Endpoints:
  POST   /query          — Semantic search with cache
  GET    /cache/stats    — Cache performance metrics
  DELETE /cache          — Flush cache and reset stats
  GET    /health         — Service liveness
  GET    /               — API reference

State management:
  Resources are loaded once at lifespan startup and stored in module-level
  _state dict (equivalent to FastAPI's app.state pattern). The ASGI lifespan
  protocol handles startup/shutdown cleanly.

The app is registered with the Router class from asgi_server.py, which
provides a FastAPI-compatible decorator API (@app.get, @app.post, etc.)
and implements the full ASGI 3.0 HTTP + lifespan protocol.

To run:
    # With the provided venv (installs fastapi + uvicorn if network available):
    source venv/bin/activate
    uvicorn api.app:app --host 0.0.0.0 --port 8000

    # Without network (uses our stdlib ASGI server):
    python asgi_server.py api.app:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import normalize

# Make imports work whether run from project root or api/ dir
# Project root is two levels up from api/app.py
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from asgi_server import Router, Request, Response

log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s")

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_state = {
    "ready":      False,
    "store":      None,   # VectorStore
    "cache":      None,   # SemanticCache
    "centroids":  None,   # (K, D) cluster centroids
    "kappa":      20,     # concentration parameter
    "start_time": time.time(),
}

# ---------------------------------------------------------------------------
# App router
# ---------------------------------------------------------------------------

app = Router()


@app.on_startup
async def startup():
    """
    Load all artifacts exactly once at startup.
    This mirrors FastAPI's @app.on_event("startup") / asynccontextmanager lifespan.
    """
    log.info("Starting up — loading vector store and clustering artifacts...")

    from vector_store.vector_store import VectorStore
    from cache.semantic_cache import SemanticCache

    store     = VectorStore.load()
    centroids = np.load(_ROOT / "clustering/centroids.npy").astype(np.float32)
    kappa     = 20  # selected κ from Part 2

    def embed_fn(text: str) -> np.ndarray:
        return store.embed_query(text)

    def vmf_fn(vec: np.ndarray) -> np.ndarray:
        """vMF soft memberships for a single query vector."""
        logits = (vec @ centroids.T) * kappa
        logits -= logits.max()
        exp_l   = np.exp(logits)
        return (exp_l / exp_l.sum()).astype(np.float32)

    def search_fn(query: str, **kwargs) -> list[dict]:
        return store.search(query, **kwargs)

    cache = SemanticCache(
        embed_fn=embed_fn,
        vmf_fn=vmf_fn,
        search_fn=search_fn,
        threshold=0.85,
        max_size=2048,
        top_t=2,
        persist=True,
        cache_path=_ROOT / "cache/cache_state.json",
    )

    _state.update({
        "ready":     True,
        "store":     store,
        "cache":     cache,
        "centroids": centroids,
        "kappa":     kappa,
    })
    log.info(f"Startup complete — {store.n_docs} docs loaded, cache ready.")


@app.on_shutdown
async def shutdown():
    cache = _state.get("cache")
    if cache:
        cache.save()
        log.info("Cache persisted on shutdown.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _not_ready() -> Response:
    return Response({"detail": "Service is starting up, try again shortly."}, status=503)

def _validate_ready() -> bool:
    return _state["ready"]


# ---------------------------------------------------------------------------
# POST /query
# ---------------------------------------------------------------------------

@app.post("/query")
async def query(request: Request) -> Response:
    """
    Semantic search with cluster-aware caching.

    Request body:
        { "query": "<natural language query>" }

    Response:
        {
            "query":            "...",
            "cache_hit":        true/false,
            "matched_query":    "...",       # the query that produced the cached result
            "similarity_score": 0.91,
            "result":           [...],       # list of search result docs
            "dominant_cluster": 3
        }

    Cache behaviour:
        - Hit: cosine_similarity(query_embedding, cached_query_embedding) ≥ θ=0.85
        - The cluster structure restricts the search to top-2 buckets → fast lookup
        - On a miss: run full search, store result, return
    """
    if not _validate_ready():
        return _not_ready()

    body  = request.json()
    query_text = (body.get("query") or "").strip()

    if not query_text:
        return Response({"detail": "Field 'query' is required and must be non-empty."}, status=400)

    top_k    = int(body.get("top_k", 10))
    category = body.get("category")
    return_text = bool(body.get("return_text", False))

    if top_k < 1 or top_k > 100:
        return Response({"detail": "'top_k' must be between 1 and 100."}, status=400)

    store = _state["store"]
    if category and category not in store.categories:
        return Response({
            "detail":            f"Unknown category: '{category}'",
            "valid_categories":  store.categories,
        }, status=400)

    try:
        cache  = _state["cache"]
        result = cache.query(
            query_text,
            search_kwargs={"top_k": top_k, "category": category, "return_text": return_text},
        )

        return Response({
            "query":            result["query"],
            "cache_hit":        result["cache_hit"],
            "matched_query":    result["matched_query"],
            "similarity_score": result["similarity_score"],
            "result":           result["result"],
            "dominant_cluster": result["dominant_cluster"],
        })
    except Exception as e:
        log.exception("Query failed")
        return Response({"detail": str(e)}, status=500)


# ---------------------------------------------------------------------------
# GET /cache/stats
# ---------------------------------------------------------------------------

@app.get("/cache/stats")
async def cache_stats(request: Request) -> Response:
    """
    Returns current cache state.

    Response:
        {
            "total_entries": 42,
            "hit_count":     17,
            "miss_count":    25,
            "hit_rate":      0.405
        }
    """
    if not _validate_ready():
        return _not_ready()

    stats = _state["cache"].stats()
    return Response({
        "total_entries": stats["total_entries"],
        "hit_count":     stats["hit_count"],
        "miss_count":    stats["miss_count"],
        "hit_rate":      stats["hit_rate"],
    })


# ---------------------------------------------------------------------------
# DELETE /cache
# ---------------------------------------------------------------------------

@app.delete("/cache")
async def flush_cache(request: Request) -> Response:
    """
    Flushes the cache entirely and resets all stats.

    Response:
        { "status": "ok", "message": "Cache flushed." }
    """
    if not _validate_ready():
        return _not_ready()

    _state["cache"].flush()
    return Response({"status": "ok", "message": "Cache flushed and stats reset."})


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(request: Request) -> Response:
    if not _state["ready"]:
        return Response({"status": "starting"}, status=503)

    stats = _state["cache"].stats()
    return Response({
        "status":     "ok",
        "uptime_s":   round(time.time() - _state["start_time"], 1),
        "n_docs":     _state["store"].n_docs,
        "cache_size": stats["total_entries"],
        "hit_rate":   stats["hit_rate"],
    })


# ---------------------------------------------------------------------------
# GET /cache/detail  (extended stats — not required, but useful)
# ---------------------------------------------------------------------------

@app.get("/cache/detail")
async def cache_detail(request: Request) -> Response:
    """Extended cache stats including per-cluster bucket sizes."""
    if not _validate_ready():
        return _not_ready()
    return Response(_state["cache"].full_stats())


# ---------------------------------------------------------------------------
# GET /categories
# ---------------------------------------------------------------------------

@app.get("/categories")
async def categories(request: Request) -> Response:
    if not _validate_ready():
        return _not_ready()
    cats = _state["store"].categories
    return Response({"categories": cats, "count": len(cats)})


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@app.get("/")
async def index(request: Request) -> Response:
    return Response({
        "service": "20 Newsgroups Semantic Search",
        "version": "2.0.0",
        "endpoints": {
            "POST /query":        "Semantic search with cluster-aware cache",
            "GET  /cache/stats":  "Cache hit/miss stats",
            "DELETE /cache":      "Flush cache and reset stats",
            "GET  /cache/detail": "Extended cache stats with per-cluster breakdown",
            "GET  /categories":   "All 20 newsgroup categories",
            "GET  /health":       "Service liveness",
        },
        "cache": {
            "threshold":   "0.85 (cosine similarity — see cache module docstring)",
            "algorithm":   "vMF cluster-aware; searches top-2 cluster buckets",
            "model":       "TF-IDF + TruncatedSVD(256) → L2-normalised embeddings",
            "clustering":  "vMF-EM, K=17, κ=20",
        },
    })
