"""
Compare TCI against other intermediate scoring methods.

Baselines:
  1. TCI (ours): Per-document k-means centroids → approximate MaxSim
  2. PLAID-style: Corpus-level k-means centroids → approximate MaxSim
  3. Token pooling (mean): Group consecutive tokens → mean pool → MaxSim
  4. Token pooling (max): Group consecutive tokens → max pool → MaxSim
  5. Random token subset: Random K tokens per document → exact MaxSim on subset
  6. Uniform token subset: Evenly-spaced K tokens per document → exact MaxSim
  7. FDE (baseline): SimHash bucket averaging → dot product

All methods produce a score for each (query, document) pair.
We measure: scoring error vs true Chamfer, inversion rate, and retrieval recall.

Usage:
  python run_baselines.py --embeddings-dir data/fiqa_colbertv2
  python run_baselines.py --embeddings-dir data/vidore_v3_finance_colpali
"""

import argparse
import json
import os
import time
import numpy as np
from tqdm import tqdm


def load_embeddings(embeddings_dir):
    corpus_flat = np.load(os.path.join(embeddings_dir, "corpus_flat.npy"))
    corpus_lengths = np.load(os.path.join(embeddings_dir, "corpus_lengths.npy"))
    query_flat = np.load(os.path.join(embeddings_dir, "query_flat.npy"))
    query_lengths = np.load(os.path.join(embeddings_dir, "query_lengths.npy"))

    with open(os.path.join(embeddings_dir, "qrels.json")) as f:
        qrels = json.load(f)

    corpus_tokens = []
    offset = 0
    for l in corpus_lengths:
        corpus_tokens.append(corpus_flat[offset:offset+l])
        offset += l

    query_tokens = []
    offset = 0
    for l in query_lengths:
        query_tokens.append(query_flat[offset:offset+l])
        offset += l

    ground_truth = qrels["ground_truth"]
    dataset_name = os.path.basename(embeddings_dir.rstrip("/"))

    return corpus_tokens, query_tokens, ground_truth, dataset_name, corpus_lengths


def chamfer_score(q_tokens, d_tokens):
    sim = q_tokens @ d_tokens.T
    return float(sim.max(axis=1).sum())


def encode_fde(doc_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = doc_tokens.shape[1]
    total_dim = R * 2 * dim
    fde = np.zeros(total_dim, dtype=np.float32)
    n_tokens = len(doc_tokens)
    for r in range(R):
        assignments = rng.randint(0, 2, size=n_tokens)
        for b in range(2):
            mask = assignments == b
            if mask.any():
                bucket_mean = doc_tokens[mask].mean(axis=0)
                start = (r * 2 + b) * dim
                fde[start:start+dim] = bucket_mean
    return fde


def encode_query_fde(query_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = query_tokens.shape[1]
    total_dim = R * 2 * dim
    fde = np.zeros(total_dim, dtype=np.float32)
    n_tokens = len(query_tokens)
    for r in range(R):
        assignments = rng.randint(0, 2, size=n_tokens)
        for b in range(2):
            mask = assignments == b
            if mask.any():
                bucket_sum = query_tokens[mask].sum(axis=0)
                start = (r * 2 + b) * dim
                fde[start:start+dim] = bucket_sum / R
    return fde


# ================================================================
# INTERMEDIATE SCORING METHODS
# ================================================================

def build_tci_index(corpus_tokens, K=32):
    """TCI: Per-document k-means clustering."""
    from sklearn.cluster import KMeans
    centroids_list = []
    for tokens in tqdm(corpus_tokens, desc=f"Building TCI-{K}"):
        n = len(tokens)
        k = min(K, n)
        if k < 2:
            centroids_list.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        centroids_list.append(km.cluster_centers_.astype(np.float32))
    return centroids_list


def build_plaid_index(corpus_tokens, K=32, n_sample=50000):
    """PLAID-style: Corpus-level k-means centroids."""
    from sklearn.cluster import MiniBatchKMeans

    # Sample tokens from corpus for clustering
    all_tokens = []
    rng = np.random.RandomState(42)
    for tokens in corpus_tokens:
        if len(all_tokens) < n_sample:
            idx = rng.choice(len(tokens), size=min(10, len(tokens)), replace=False)
            all_tokens.extend(tokens[idx])

    all_tokens = np.array(all_tokens[:n_sample])
    print(f"  PLAID: clustering {len(all_tokens)} sampled tokens into {K} centroids...")

    km = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=100, random_state=42,
                         batch_size=min(10000, len(all_tokens)))
    km.fit(all_tokens)
    corpus_centroids = km.cluster_centers_.astype(np.float32)

    # For each document, find nearest centroid for each token
    # Score = sum over query tokens of max similarity to document's assigned centroids
    doc_centroid_ids = []
    for tokens in tqdm(corpus_tokens, desc=f"PLAID assignments"):
        assignments = km.predict(tokens)
        unique_centroids = np.unique(assignments)
        doc_centroid_ids.append(unique_centroids)

    return corpus_centroids, doc_centroid_ids


def build_token_pool_index(corpus_tokens, K=32, method="mean"):
    """Token pooling: Group consecutive tokens and pool."""
    pooled_list = []
    for tokens in corpus_tokens:
        n = len(tokens)
        if n <= K:
            pooled_list.append(tokens.copy())
            continue

        group_size = n // K
        pooled = np.zeros((K, tokens.shape[1]), dtype=np.float32)
        for i in range(K):
            start = i * group_size
            end = start + group_size if i < K - 1 else n
            if method == "mean":
                pooled[i] = tokens[start:end].mean(axis=0)
            elif method == "max":
                pooled[i] = tokens[start:end].max(axis=0)
        pooled_list.append(pooled)
    return pooled_list


def build_random_subset_index(corpus_tokens, K=32, seed=42):
    """Random token subset: Keep K random tokens per document."""
    rng = np.random.RandomState(seed)
    subset_list = []
    for tokens in corpus_tokens:
        n = len(tokens)
        if n <= K:
            subset_list.append(tokens.copy())
        else:
            idx = rng.choice(n, size=K, replace=False)
            subset_list.append(tokens[idx].copy())
    return subset_list


def build_uniform_subset_index(corpus_tokens, K=32):
    """Uniform token subset: Keep K evenly-spaced tokens per document."""
    subset_list = []
    for tokens in corpus_tokens:
        n = len(tokens)
        if n <= K:
            subset_list.append(tokens.copy())
        else:
            idx = np.linspace(0, n - 1, K, dtype=int)
            subset_list.append(tokens[idx].copy())
    return subset_list


def score_maxsim(q_tokens, d_repr):
    """MaxSim score between query tokens and document representation."""
    sim = q_tokens @ d_repr.T
    return float(sim.max(axis=1).sum())


def score_plaid(q_tokens, corpus_centroids, doc_centroid_ids):
    """PLAID-style scoring: MaxSim using only document's assigned centroids."""
    doc_centroids = corpus_centroids[doc_centroid_ids]
    sim = q_tokens @ doc_centroids.T
    return float(sim.max(axis=1).sum())


def per_query_recall(ranked_indices, ground_truth_set, k):
    if not ground_truth_set:
        return None
    hits = sum(1 for idx in ranked_indices[:k] if idx in ground_truth_set)
    return hits / len(ground_truth_set)


def run(args):
    print("=" * 70)
    print("BASELINE COMPARISON: TCI vs Other Intermediate Scoring Methods")
    print("=" * 70)

    corpus_tokens, query_tokens, ground_truth, dataset_name, corpus_lengths = \
        load_embeddings(args.embeddings_dir)

    n_docs = len(corpus_tokens)
    n_queries = len(query_tokens)
    avg_tokens = float(np.mean(corpus_lengths))
    K = args.K

    valid_queries = [i for i in range(n_queries) if len(ground_truth[i]) > 0]
    n_eval = min(args.n_queries, len(valid_queries))
    eval_queries = valid_queries[:n_eval]

    print(f"  {n_docs} docs (avg {avg_tokens:.0f} tok), {n_queries} queries")
    print(f"  Evaluating {n_eval} queries, K={K} for all methods")

    # ================================================================
    # Build all indices
    # ================================================================
    print(f"\n{'='*70}")
    print("BUILDING INDICES")
    print(f"{'='*70}")

    # FDE
    print("\nEncoding FDEs...")
    doc_fdes = np.array([encode_fde(d, R=10) for d in corpus_tokens])
    query_fdes = np.array([encode_query_fde(q, R=10) for q in query_tokens])

    # TCI (ours)
    print(f"\nBuilding TCI-{K}...")
    t0 = time.time()
    tci_index = build_tci_index(corpus_tokens, K=K)
    tci_time = time.time() - t0
    print(f"  Built in {tci_time:.1f}s")

    # PLAID-style corpus centroids
    print(f"\nBuilding PLAID-{K}...")
    t0 = time.time()
    plaid_centroids, plaid_doc_ids = build_plaid_index(corpus_tokens, K=K)
    plaid_time = time.time() - t0
    print(f"  Built in {plaid_time:.1f}s")

    # Token pooling (mean)
    print(f"\nBuilding MeanPool-{K}...")
    t0 = time.time()
    meanpool_index = build_token_pool_index(corpus_tokens, K=K, method="mean")
    meanpool_time = time.time() - t0
    print(f"  Built in {meanpool_time:.1f}s")

    # Token pooling (max)
    print(f"\nBuilding MaxPool-{K}...")
    t0 = time.time()
    maxpool_index = build_token_pool_index(corpus_tokens, K=K, method="max")
    maxpool_time = time.time() - t0
    print(f"  Built in {maxpool_time:.1f}s")

    # Random subset
    print(f"\nBuilding RandomSubset-{K}...")
    t0 = time.time()
    random_index = build_random_subset_index(corpus_tokens, K=K)
    random_time = time.time() - t0
    print(f"  Built in {random_time:.1f}s")

    # Uniform subset
    print(f"\nBuilding UniformSubset-{K}...")
    t0 = time.time()
    uniform_index = build_uniform_subset_index(corpus_tokens, K=K)
    uniform_time = time.time() - t0
    print(f"  Built in {uniform_time:.1f}s")

    # ================================================================
    # Evaluate scoring accuracy and retrieval
    # ================================================================
    print(f"\n{'='*70}")
    print(f"EVALUATING ({n_eval} queries)")
    print(f"{'='*70}")

    methods = {
        "FDE": {"type": "fde"},
        f"TCI-{K} (ours)": {"type": "maxsim", "index": tci_index},
        f"PLAID-{K}": {"type": "plaid", "centroids": plaid_centroids, "doc_ids": plaid_doc_ids},
        f"MeanPool-{K}": {"type": "maxsim", "index": meanpool_index},
        f"MaxPool-{K}": {"type": "maxsim", "index": maxpool_index},
        f"RandomSub-{K}": {"type": "maxsim", "index": random_index},
        f"UniformSub-{K}": {"type": "maxsim", "index": uniform_index},
    }

    W = args.rerank_budget  # Chamfer rerank budget
    W_prime = args.fde_candidates  # FDE candidates

    results = {}

    for method_name, method_info in methods.items():
        print(f"\n--- {method_name} ---")

        # Per-query metrics
        scoring_errors_rel = []
        scoring_errors_neg = []
        inversions = 0
        total_pairs = 0
        r10_list = []
        r100_list = []

        for qi in tqdm(eval_queries, desc=method_name):
            gt_set = set(ground_truth[qi])
            q_tok = query_tokens[qi]

            # Get FDE top candidates
            fde_scores = doc_fdes @ query_fdes[qi]
            fde_top = np.argsort(-fde_scores)[:W_prime]

            # Score candidates with this method
            method_scores = []
            true_scores = []

            for di in fde_top:
                true_s = chamfer_score(q_tok, corpus_tokens[di])
                true_scores.append(true_s)

                if method_info["type"] == "fde":
                    approx_s = float(fde_scores[di])
                elif method_info["type"] == "plaid":
                    approx_s = score_plaid(q_tok, method_info["centroids"],
                                           method_info["doc_ids"][di])
                elif method_info["type"] == "maxsim":
                    approx_s = score_maxsim(q_tok, method_info["index"][di])
                else:
                    approx_s = 0.0

                method_scores.append(approx_s)

            method_scores = np.array(method_scores)
            true_scores = np.array(true_scores)

            # Scoring error
            for j, di in enumerate(fde_top):
                if true_scores[j] > 0:
                    rel_error = abs(method_scores[j] - true_scores[j]) / abs(true_scores[j])
                    if di in gt_set:
                        scoring_errors_rel.append(rel_error)
                    else:
                        scoring_errors_neg.append(rel_error)

            # Inversions: count pairs where method ranking disagrees with true ranking
            # Sample relevant vs negative pairs
            rel_indices = [j for j, di in enumerate(fde_top) if di in gt_set]
            neg_indices = [j for j, di in enumerate(fde_top) if di not in gt_set]

            for ri in rel_indices[:5]:  # limit pairs per query
                for ni in neg_indices[:10]:
                    if true_scores[ri] > true_scores[ni]:
                        total_pairs += 1
                        if method_scores[ri] <= method_scores[ni]:
                            inversions += 1

            # Retrieval: rescore top-W_prime, take top-W, Chamfer rerank
            rescore_order = np.argsort(-method_scores)[:W]
            rerank_candidates = fde_top[rescore_order]

            # Chamfer rerank
            chamfer_results = []
            for di in rerank_candidates:
                s = chamfer_score(q_tok, corpus_tokens[di])
                chamfer_results.append((di, s))
            chamfer_results.sort(key=lambda x: -x[1])
            ranked = [x[0] for x in chamfer_results]

            r10_list.append(per_query_recall(ranked, gt_set, 10))
            r100_list.append(per_query_recall(ranked, gt_set, min(100, W)))

        # Aggregate
        inv_rate = inversions / total_pairs if total_pairs > 0 else 0
        mean_r10 = np.mean(r10_list)
        mean_r100 = np.mean(r100_list)
        mean_err_rel = np.mean(scoring_errors_rel) if scoring_errors_rel else 0
        mean_err_neg = np.mean(scoring_errors_neg) if scoring_errors_neg else 0

        results[method_name] = {
            "scoring_error_relevant": float(mean_err_rel),
            "scoring_error_negative": float(mean_err_neg),
            "inversion_rate": float(inv_rate),
            "inversions": inversions,
            "total_pairs": total_pairs,
            "R@10": float(mean_r10),
            "R@100": float(mean_r100),
        }

        print(f"  Scoring error (rel): {mean_err_rel:.3f}")
        print(f"  Inversion rate: {inv_rate:.4f} ({inversions}/{total_pairs})")
        print(f"  R@10: {mean_r10:.4f}  R@100: {mean_r100:.4f}")

    # ================================================================
    # Summary table
    # ================================================================
    print(f"\n{'='*70}")
    print(f"SUMMARY — {dataset_name} ({n_docs} docs, {n_eval} queries, K={K})")
    print(f"FDE top-{W_prime} → Method rescore top-{W} → Chamfer rerank")
    print(f"{'='*70}")
    print(f"  {'Method':<25} | {'Error(rel)':>10} | {'Error(neg)':>10} | {'Inv rate':>10} | {'R@10':>7} | {'R@100':>7}")
    print(f"  {'-'*85}")

    # Sort by R@10
    for name in sorted(results.keys(), key=lambda x: -results[x]["R@10"]):
        r = results[name]
        marker = " ← OURS" if "ours" in name else ""
        print(f"  {name:<25} | {r['scoring_error_relevant']:>10.3f} | {r['scoring_error_negative']:>10.3f} | {r['inversion_rate']:>10.4f} | {r['R@10']:>7.4f} | {r['R@100']:>7.4f}{marker}")

    # Save
    out_dir = f"./results/baselines_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "baseline_results.json"), "w") as f:
        json.dump({
            "dataset": dataset_name,
            "n_docs": n_docs,
            "n_queries": n_eval,
            "K": K,
            "rerank_budget": W,
            "fde_candidates": W_prime,
            "results": results,
        }, f, indent=2)

    print(f"\nSaved to {out_dir}/baseline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--K", type=int, default=32, help="Number of centroids/pools/subsets")
    parser.add_argument("--rerank-budget", type=int, default=200, help="Chamfer rerank budget W")
    parser.add_argument("--fde-candidates", type=int, default=1000, help="FDE candidates W'")
    parser.add_argument("--n-queries", type=int, default=200, help="Number of queries to evaluate")
    args = parser.parse_args()
    run(args)
