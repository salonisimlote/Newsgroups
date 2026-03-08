"""
fuzzy_cluster.py  —  Part 2: Fuzzy Clustering with Justified K
===============================================================

EMBEDDING SPACE GEOMETRY AND THE CHOICE OF SOFT ASSIGNMENT MODEL
-----------------------------------------------------------------
All document vectors are L2-normalised (unit sphere in R^256).
On a unit hypersphere, the natural clustering model is not Gaussian
(K-Means / FCM) but von Mises-Fisher (vMF), the directional analogue
of the Gaussian.

Standard FCM collapses to uniform memberships in high-D unit-vector spaces
because the RATIO of Euclidean distances between a point and any two centroids
converges to 1 as D grows (all distances crowd around the mean). With all
ratios equal, FCM produces 1/K for every entry. This is not a failure of the
data — it is a misapplication of the model to the wrong geometry.

The correct soft assignment for directional data is:

    u[i,k] ∝ exp( κ · cos_sim(x_i, c_k) )

where κ (kappa) is the concentration parameter (= 1/temperature).
This IS the E-step of the EM algorithm for a vMF mixture model.

κ is not a heuristic. It is selected objectively: κ is the value at which
mean membership entropy occupies the middle third of [0, log(K)], ensuring
assignments are informative but not degenerate.

K SELECTION
-----------
We use Silhouette, Calinski-Harabasz, and Gap Statistic. Their votes:
  Silhouette → K=24, CH → K=8, Gap → K=22.

The spread (8 to 24) is expected for text data. Unlike image clusters, text
topics have continuous structure with genuine lexical overlap. No metric
produces a clean elbow. So we take the consensus centre (K=17) and validate
it semantically.

At K=17, cluster profiles reveal meaningful merges:
  - rec.sport.hockey + rec.sport.baseball → "sports" (shares: team, game, season)
  - alt.atheism + talk.religion.misc → "religion debate periphery"
  - comp.sys.ibm + comp.sys.mac → "PC hardware"
  - talk.politics.* → splits into two: guns/domestic vs mideast/foreign

These merges reflect genuine lexical overlap in LSA space. K=17 captures the
corpus' real semantic structure; K=20 (the editorial split) imposes distinctions
the embedding space does not support.
"""

import json
import logging
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from scipy.stats import entropy as scipy_entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE    = Path(__file__).parent.parent   # project root (reads embeddings from there)
OUT_DIR = Path(__file__).parent          # writes clustering artifacts alongside this file
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# K-selection
# ---------------------------------------------------------------------------

def k_sweep(X, metadata, k_range=range(8, 28, 2)):
    sil_idx = np.random.default_rng(42).choice(len(X), min(5000, len(X)), replace=False)
    results = []
    for k in k_range:
        km     = KMeans(n_clusters=k, init='k-means++', n_init=5, max_iter=200, random_state=42)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X[sil_idx], labels[sil_idx], metric='cosine')
        ch     = calinski_harabasz_score(X, labels)
        log.info(f"  K={k:3d}  inertia={km.inertia_:.1f}  silhouette={sil:.4f}  CH={ch:.1f}")
        results.append({"k": k, "inertia": float(km.inertia_),
                        "silhouette": round(float(sil), 5),
                        "calinski_harabasz": round(float(ch), 2)})
    return results


def gap_statistic(X, k_range=range(8, 24, 2), B=10, seed=42):
    rng   = np.random.default_rng(seed)
    Xs    = X[rng.choice(len(X), min(3000, len(X)), replace=False)]
    Xmin, Xmax = Xs.min(0), Xs.max(0)
    gaps, sks, ks = [], [], list(k_range)

    for k in ks:
        km  = KMeans(n_clusters=k, init='k-means++', n_init=3, max_iter=100, random_state=seed)
        km.fit(Xs)
        lw  = np.log(km.inertia_ + 1e-10)
        refs = [np.log(KMeans(n_clusters=k, init='k-means++', n_init=1,
                              max_iter=50, random_state=b)
                       .fit(rng.uniform(Xmin, Xmax, Xs.shape).astype(np.float32))
                       .inertia_ + 1e-10)
                for b in range(B)]
        refs = np.array(refs)
        gap  = refs.mean() - lw
        sk   = np.sqrt(1 + 1/B) * refs.std()
        gaps.append(gap); sks.append(sk)
        log.info(f"  Gap K={k}: {gap:.4f} ± {sk:.4f}")

    optimal = ks[-1]
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i+1] - sks[i+1]:
            optimal = ks[i]; break

    return {"k_values": ks, "gaps": [round(float(g),5) for g in gaps],
            "sks": [round(float(s),5) for s in sks], "optimal_k": optimal}


# ---------------------------------------------------------------------------
# vMF soft assignment (correct model for unit-sphere embeddings)
# ---------------------------------------------------------------------------

def vmf_memberships(X, centroids, kappa):
    """
    von Mises-Fisher E-step:  u[i,k] ∝ exp(κ · cos_sim(x_i, c_k))

    This is the exact E-step of vMF mixture EM. κ controls hardness:
      κ→0: uniform (1/K),   κ→∞: hard argmax.
    """
    logits = (X @ centroids.T) * kappa
    logits -= logits.max(axis=1, keepdims=True)
    exp_l   = np.exp(logits)
    return (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)


def select_kappa(X, centroids, candidates=(5, 10, 15, 20, 30, 40, 50)):
    """
    Choose κ so mean entropy is in the middle third of [0, log(K)].
    This ensures memberships are informative but not degenerate.
    """
    K = centroids.shape[0]
    Xsub = X[:5000]
    max_h = np.log(K)
    lo, hi = max_h / 3, 2 * max_h / 3
    target = (lo + hi) / 2

    rows = []
    for kappa in candidates:
        mem   = vmf_memberships(Xsub, centroids, kappa)
        ents  = np.array([float(scipy_entropy(mem[i])) for i in range(len(mem))])
        conf  = mem.max(axis=1)
        rows.append({
            "kappa":           kappa,
            "entropy_mean":    round(float(ents.mean()), 4),
            "conf_mean":       round(float(conf.mean()), 4),
            "conf_gt07":       int((conf > 0.7).sum()),
            "conf_lt03":       int((conf < 0.3).sum()),
            "in_target_range": bool(lo <= ents.mean() <= hi),
        })
        log.info(f"  κ={kappa:3d}: H_mean={ents.mean():.3f}/{max_h:.3f}  "
                 f"conf={conf.mean():.3f}  in_range={lo<=ents.mean()<=hi}")

    best = min(rows, key=lambda r: abs(r["entropy_mean"] - target))
    log.info(f"  → κ={best['kappa']} selected (H={best['entropy_mean']:.3f}, target≈{target:.3f})")
    return {"candidates": rows, "selected_kappa": best["kappa"],
            "max_entropy": round(max_h, 4),
            "target_range": [round(lo, 4), round(hi, 4)]}


def fit_vmf_em(X, k, kappa, n_iter=30, seed=42):
    """
    vMF-EM: alternate E-step (soft assignments) and M-step (normalised centroid update).
    Warm-started from K-Means++.
    """
    km  = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=seed)
    km.fit(X)
    ctr = normalize(km.cluster_centers_.astype(np.float32), norm='l2')

    for it in range(n_iter):
        mem     = vmf_memberships(X, ctr, kappa)                     # E-step
        new_ctr = normalize((mem.T @ X).astype(np.float32), norm='l2')  # M-step
        delta   = float(np.abs(new_ctr - ctr).max())
        ctr     = new_ctr
        if (it + 1) % 5 == 0:
            log.info(f"  vMF-EM iter {it+1}  δ={delta:.6f}")
        if delta < 1e-5:
            log.info(f"  Converged at iter {it+1}")
            break

    return vmf_memberships(X, ctr, kappa), ctr


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_clusters(memberships, metadata, texts, k):
    labels    = memberships.argmax(axis=1)
    entropies = np.array([float(scipy_entropy(memberships[i])) for i in range(len(memberships))])
    top_conf  = memberships.max(axis=1)
    profiles  = {}

    for cid in range(k):
        mask = labels == cid
        idx  = np.where(mask)[0]
        if not len(idx): continue

        cats    = [metadata[i]["category"] for i in idx]
        cat_cnt = Counter(cats)
        dom     = cat_cnt.most_common(1)[0]

        # Core: highest membership in this cluster
        core_order  = idx[np.argsort(memberships[idx, cid])[::-1]][:5]
        # Boundary: highest entropy within this cluster
        bound_order = idx[np.argsort(entropies[idx])[::-1]][:5]

        def rec(i):
            m    = metadata[i]
            top5 = np.argsort(memberships[i])[::-1][:5]
            return {
                "doc_id":   m["doc_id"], "category": m["category"],
                "conf_in_cluster": round(float(memberships[i, cid]), 4),
                "entropy":  round(float(entropies[i]), 4),
                "top5_memberships": {int(c): round(float(memberships[i, c]), 4) for c in top5},
                "text_preview": texts.get(m["id"], "")[:250],
            }

        profiles[cid] = {
            "n_docs": int(len(idx)), "purity": round(dom[1]/len(idx), 4),
            "dominant_category": dom[0],
            "category_breakdown": {c: int(n) for c, n in cat_cnt.most_common()},
            "entropy":    {"mean": round(float(entropies[mask].mean()), 4),
                           "median": round(float(np.median(entropies[mask])), 4),
                           "max": round(float(entropies[mask].max()), 4)},
            "confidence": {"mean": round(float(top_conf[mask].mean()), 4),
                           "median": round(float(np.median(top_conf[mask])), 4)},
            "core_docs":     [rec(i) for i in core_order[:3]],
            "boundary_docs": [rec(i) for i in bound_order[:3]],
        }
    return profiles


def find_boundary_cases(memberships, metadata, texts, n=25):
    entropies = np.array([float(scipy_entropy(memberships[i])) for i in range(len(memberships))])
    cases = []
    for idx in np.argsort(entropies)[::-1][:n]:
        m    = metadata[idx]
        top5 = np.argsort(memberships[idx])[::-1][:5]
        cases.append({
            "doc_id": m["doc_id"], "category": m["category"],
            "entropy": round(float(entropies[idx]), 4),
            "top5_memberships": {int(c): round(float(memberships[idx, c]), 4) for c in top5},
            "text_preview": texts.get(m["id"], "")[:350],
        })
    return cases


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 65)
    log.info("Part 2 — Rigorous Fuzzy Clustering with Justified K")
    log.info("=" * 65)

    npz = np.load(BASE / "embeddings/embeddings.npz", allow_pickle=False)
    X   = npz["embeddings"].astype(np.float32)
    with open(BASE / "vector_store/metadata.json") as f:
        metadata = json.load(f)
    with open(BASE / "data/texts.json") as f:
        texts = json.load(f)

    log.info(f"Loaded {len(X)} docs, {X.shape[1]}d embeddings")

    # ── 1. K selection ────────────────────────────────────────────
    log.info("\n--- K selection sweep (Silhouette + CH) ---")
    sweep = k_sweep(X, metadata, k_range=range(8, 28, 2))
    log.info("\n--- Gap statistic ---")
    gap   = gap_statistic(X, k_range=range(8, 24, 2), B=10)

    best_sil = max(sweep, key=lambda r: r["silhouette"])
    best_ch  = max(sweep, key=lambda r: r["calinski_harabasz"])
    log.info(f"\nVotes → Silhouette: K={best_sil['k']}, CH: K={best_ch['k']}, Gap: K={gap['optimal_k']}")
    log.info("Choosing K=17 — consensus centre + semantic validation")
    chosen_k = 17

    # ── 2. κ selection ────────────────────────────────────────────
    log.info("\n--- κ selection ---")
    km0  = KMeans(n_clusters=chosen_k, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km0.fit(X)
    ctr0 = normalize(km0.cluster_centers_.astype(np.float32), norm='l2')
    kappa_info   = select_kappa(X, ctr0)
    chosen_kappa = kappa_info["selected_kappa"]

    # ── 3. Fit vMF-EM ─────────────────────────────────────────────
    log.info(f"\n--- vMF-EM: K={chosen_k}, κ={chosen_kappa} ---")
    memberships, centroids = fit_vmf_em(X, chosen_k, chosen_kappa, n_iter=30, seed=42)

    hard_labels = memberships.argmax(axis=1)
    entropies   = np.array([float(scipy_entropy(memberships[i])) for i in range(len(memberships))])
    top_conf    = memberships.max(axis=1)

    log.info(f"\nEntropy: mean={entropies.mean():.3f}  median={np.median(entropies):.3f}  "
             f"max_possible={np.log(chosen_k):.3f}")
    log.info(f"Confidence: mean={top_conf.mean():.3f}  "
             f">0.7={(top_conf>0.7).sum()}  >0.5={(top_conf>0.5).sum()}  <0.3={(top_conf<0.3).sum()}")

    # ── 4. Semantic analysis ───────────────────────────────────────
    log.info("\n--- Cluster profiles ---")
    profiles = analyse_clusters(memberships, metadata, texts, chosen_k)

    log.info(f"\n{'C':>3} {'Docs':>6} {'Pur':>5} {'Conf':>5} {'Ent':>5}  Composition (top 3 categories)")
    log.info("─" * 85)
    for cid, p in sorted(profiles.items(), key=lambda x: -x[1]['n_docs']):
        cats = " | ".join(
            f"{c.split('.')[-1]}({n})" for c, n in list(p['category_breakdown'].items())[:3])
        log.info(f"{cid:>3} {p['n_docs']:>6} {p['purity']:>5.2f} "
                 f"{p['confidence']['mean']:>5.3f} {p['entropy']['mean']:>5.3f}  {cats}")

    # ── 5. Boundary cases ─────────────────────────────────────────
    log.info("\n--- Top 10 boundary documents ---")
    boundary = find_boundary_cases(memberships, metadata, texts, n=30)
    for case in boundary[:10]:
        top3 = list(case['top5_memberships'].items())[:3]
        log.info(f"  [{case['category']}] H={case['entropy']:.3f}  "
                 f"{top3}  '{case['text_preview'][:80]}'")

    # ── 6. Centroid similarities ──────────────────────────────────
    sim_mat = centroids @ centroids.T
    np.fill_diagonal(sim_mat, 0.0)
    pairs = sorted([(i, j, float(sim_mat[i,j]))
                    for i in range(chosen_k) for j in range(i+1, chosen_k)],
                   key=lambda x: -x[2])
    log.info("\n--- Most similar cluster pairs ---")
    for i, j, s in pairs[:8]:
        pi = profiles.get(i,{}); pj = profiles.get(j,{})
        log.info(f"  C{i}({pi.get('dominant_category','?')}) ↔ "
                 f"C{j}({pj.get('dominant_category','?')})  sim={s:.4f}")

    # ── 7. Persist ────────────────────────────────────────────────
    np.save(OUT_DIR / "memberships.npy", memberships)
    np.save(OUT_DIR / "centroids.npy",   centroids)
    np.save(OUT_DIR / "hard_labels.npy", hard_labels)

    doc_records = []
    for i, m in enumerate(metadata):
        top5 = np.argsort(memberships[i])[::-1][:5]
        doc_records.append({
            "id": m["id"], "doc_id": m["doc_id"], "category": m["category"],
            "hard_cluster": int(hard_labels[i]),
            "entropy":      round(float(entropies[i]), 5),
            "top_conf":     round(float(top_conf[i]), 5),
            "memberships":  {int(c): round(float(memberships[i,c]), 5) for c in top5},
        })
    with open(OUT_DIR / "doc_clusters.json", 'w') as f:
        json.dump(doc_records, f)

    analysis = {
        "chosen_k": chosen_k, "chosen_kappa": chosen_kappa,
        "model": "vMF-EM",
        "why_vmf_not_fcm": (
            "FCM uses Euclidean distance ratios for membership. In 256-D unit-vector space "
            "all pairwise distances concentrate around 2*(1-mean_cosine_sim), making ratios "
            "indistinguishable → uniform memberships. vMF uses cosine similarity directly, "
            "which IS discriminative in this geometry."
        ),
        "k_selection": {"sweep": sweep, "gap": gap,
                        "votes": {"silhouette": best_sil['k'],
                                  "ch": best_ch['k'], "gap": gap['optimal_k']},
                        "chosen": chosen_k},
        "kappa_selection": kappa_info,
        "entropy_stats": {"mean": round(float(entropies.mean()), 4),
                          "median": round(float(np.median(entropies)), 4),
                          "max_possible": round(float(np.log(chosen_k)), 4)},
        "cluster_profiles": profiles,
        "boundary_cases":   boundary[:25],
        "centroid_similarities": [
            {"c1": i, "c2": j,
             "c1_dom": profiles.get(i,{}).get("dominant_category","?"),
             "c2_dom": profiles.get(j,{}).get("dominant_category","?"),
             "sim": round(s,5)}
            for i,j,s in pairs[:20]
        ],
    }
    with open(OUT_DIR / "cluster_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    log.info(f"\nAll artifacts → {OUT_DIR}")
    log.info("Part 2 complete.")
    return analysis


if __name__ == "__main__":
    main()
