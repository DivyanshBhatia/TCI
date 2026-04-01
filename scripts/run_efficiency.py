"""
Efficiency benchmark: Measure wall-clock time per pipeline stage.

Measures each stage independently:
  1. FDE search (full vs cascade)
  2. TCI scoring
  3. Chamfer reranking

Reports: ops/second, total time, and "iso-recall" comparison
(what budget does each method need to reach the same recall?)

Usage:
  python run_efficiency.py --embeddings-dir data/fiqa_colbertv2
  python run_efficiency.py --embeddings-dir data/lotte_science_forum_colbertv2
"""

import argparse
import json
import os
import time
import numpy as np
from pathlib import Path


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


def chamfer_score(q_tokens, d_tokens):
    sim = q_tokens @ d_tokens.T
    return float(sim.max(axis=1).sum())


def tci_score(q_tokens, centroids):
    sim = q_tokens @ centroids.T
    return float(sim.max(axis=1).sum())


def per_query_recall(ranked_indices, ground_truth_set, k):
    if not ground_truth_set:
        return None
    hits = sum(1 for idx in ranked_indices[:k] if idx in ground_truth_set)
    return hits / len(ground_truth_set)


def run(args):
    print("=" * 70)
    print("EFFICIENCY BENCHMARK")
    print("=" * 70)

    corpus_tokens, query_tokens, ground_truth, dataset_name, corpus_lengths = \
        load_embeddings(args.embeddings_dir)

    n_docs = len(corpus_tokens)
    n_queries = len(query_tokens)
    avg_tokens = float(np.mean(corpus_lengths))

    valid_queries = [i for i in range(n_queries) if len(ground_truth[i]) > 0]
    n_eval = min(args.n_queries, len(valid_queries))
    eval_queries = valid_queries[:n_eval]

    print(f"  {n_docs} docs (avg {avg_tokens:.0f} tok), {n_queries} queries")
    print(f"  Evaluating {n_eval} queries")

    # ================================================================
    # Step 1: Encode FDEs (one-time cost, not benchmarked per-query)
    # ================================================================
    print("\nEncoding FDEs...")
    t0 = time.time()
    doc_fdes = np.array([encode_fde(d, R=10) for d in corpus_tokens])
    fde_encode_time = time.time() - t0
    print(f"  Doc FDE encoding: {fde_encode_time:.1f}s ({n_docs/fde_encode_time:.0f} docs/s)")

    query_fdes = np.array([encode_query_fde(q, R=10) for q in query_tokens])

    # Coarse FDEs (2-rep)
    doc_fdes_coarse = doc_fdes[:, :2*2*corpus_tokens[0].shape[1]]  # first 2 reps
    query_fdes_coarse = query_fdes[:, :2*2*query_tokens[0].shape[1]]

    # ================================================================
    # Step 2: Build TCI index (one-time cost)
    # ================================================================
    from sklearn.cluster import KMeans
    K = 32
    print(f"\nBuilding TCI-{K} index...")
    t0 = time.time()
    tci_centroids = []
    for tokens in corpus_tokens:
        n = len(tokens)
        k = min(K, n)
        if k < 2:
            tci_centroids.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        tci_centroids.append(km.cluster_centers_.astype(np.float32))
    tci_build_time = time.time() - t0
    print(f"  TCI index build: {tci_build_time:.1f}s ({n_docs/tci_build_time:.0f} docs/s)")

    # ================================================================
    # Step 3: Benchmark each stage
    # ================================================================
    print(f"\n{'='*70}")
    print("STAGE TIMING (per query, averaged over {n_eval} queries)")
    print(f"{'='*70}")

    # --- Stage A: FDE search (full) ---
    t0 = time.time()
    for qi in eval_queries:
        scores = doc_fdes @ query_fdes[qi]
        top_1000 = np.argsort(-scores)[:1000]
    fde_full_time = (time.time() - t0) / n_eval
    fde_full_ops = n_docs  # one dot product per doc

    # --- Stage A2: FDE search (coarse cascade) ---
    t0 = time.time()
    for qi in eval_queries:
        coarse_scores = doc_fdes_coarse @ query_fdes_coarse[qi]
        top_5000_coarse = np.argsort(-coarse_scores)[:5000]
        # Then full FDE on top-5000
        full_scores = doc_fdes[top_5000_coarse] @ query_fdes[qi]
        top_1000 = top_5000_coarse[np.argsort(-full_scores)[:1000]]
    cascade_time = (time.time() - t0) / n_eval
    cascade_ops_coarse = n_docs  # coarse search
    cascade_ops_full = 5000     # full FDE on candidates

    # --- Stage B: TCI scoring ---
    # Get FDE top-1000 first
    fde_top1000_per_query = []
    for qi in eval_queries:
        scores = doc_fdes @ query_fdes[qi]
        fde_top1000_per_query.append(np.argsort(-scores)[:1000])

    t0 = time.time()
    for idx, qi in enumerate(eval_queries):
        candidates = fde_top1000_per_query[idx]
        tci_scores = []
        for di in candidates:
            s = tci_score(query_tokens[qi], tci_centroids[di])
            tci_scores.append(s)
        tci_scores = np.array(tci_scores)
        top_200 = candidates[np.argsort(-tci_scores)[:200]]
    tci_time = (time.time() - t0) / n_eval
    tci_ops = 1000  # score 1000 candidates

    # --- Stage C: Chamfer reranking ---
    # Benchmark at different W values
    chamfer_times = {}
    for W in [50, 100, 200, 500, 1000]:
        t0 = time.time()
        for idx, qi in enumerate(eval_queries):
            candidates = fde_top1000_per_query[idx][:W]
            chamfer_scores = []
            for di in candidates:
                s = chamfer_score(query_tokens[qi], corpus_tokens[di])
                chamfer_scores.append(s)
        chamfer_times[W] = (time.time() - t0) / n_eval

    # ================================================================
    # Step 4: Recall at different budgets
    # ================================================================
    print(f"\n{'='*70}")
    print("RECALL vs CHAMFER BUDGET")
    print(f"{'='*70}")

    from tqdm import tqdm

    pipelines = {}

    for W in [50, 100, 200, 500, 1000]:
        # [A] MUVERA: FDE → Chamfer(W)
        r10_list = []
        r100_list = []
        for idx, qi in enumerate(eval_queries):
            gt_set = set(ground_truth[qi])
            candidates = fde_top1000_per_query[idx][:W]
            scores = [(di, chamfer_score(query_tokens[qi], corpus_tokens[di])) for di in candidates]
            scores.sort(key=lambda x: -x[1])
            ranked = [x[0] for x in scores]
            r10_list.append(per_query_recall(ranked, gt_set, 10))
            r100_list.append(per_query_recall(ranked, gt_set, min(100, W)))
        pipelines[f"[A] MUVERA W={W}"] = {
            "r10": np.mean(r10_list),
            "r100": np.mean(r100_list),
            "chamfer_evals": W,
            "time_ms": (fde_full_time + chamfer_times[W]) * 1000,
        }

    for W in [50, 100, 200, 500]:
        # [C] TCI: FDE(1000) → TCI(W) → Chamfer(W)
        r10_list = []
        r100_list = []
        for idx, qi in enumerate(eval_queries):
            gt_set = set(ground_truth[qi])
            candidates = fde_top1000_per_query[idx]
            # TCI rescore
            tscores = [(di, tci_score(query_tokens[qi], tci_centroids[di])) for di in candidates]
            tscores.sort(key=lambda x: -x[1])
            tci_top = [x[0] for x in tscores[:W]]
            # Chamfer
            scores = [(di, chamfer_score(query_tokens[qi], corpus_tokens[di])) for di in tci_top]
            scores.sort(key=lambda x: -x[1])
            ranked = [x[0] for x in scores]
            r10_list.append(per_query_recall(ranked, gt_set, 10))
            r100_list.append(per_query_recall(ranked, gt_set, min(100, W)))
        pipelines[f"[C] TCI W={W}"] = {
            "r10": np.mean(r10_list),
            "r100": np.mean(r100_list),
            "chamfer_evals": W,
            "time_ms": (fde_full_time + tci_time + chamfer_times[W]) * 1000,
        }

    # ================================================================
    # Step 5: Print results
    # ================================================================
    print(f"\n{'='*70}")
    print("TIMING PER STAGE")
    print(f"{'='*70}")
    print(f"  FDE search (full, {n_docs} docs):     {fde_full_time*1000:.1f} ms/query")
    print(f"  FDE search (cascade 2rep→5K→full):  {cascade_time*1000:.1f} ms/query")
    print(f"  Cascade speedup:                     {fde_full_time/cascade_time:.1f}x")
    print(f"  TCI scoring (1000 candidates):       {tci_time*1000:.1f} ms/query")
    print(f"  Chamfer rerank per doc:")
    for W in [50, 100, 200, 500, 1000]:
        per_doc = chamfer_times[W] / W * 1000
        print(f"    W={W:4d}: {chamfer_times[W]*1000:.1f} ms total ({per_doc:.2f} ms/doc)")

    print(f"\n{'='*70}")
    print("ISO-RECALL COMPARISON")
    print(f"  Same recall, fewer Chamfer evaluations = more efficient")
    print(f"{'='*70}")
    print(f"  {'Method':<30} | {'Chamfer evals':>13} | {'R@10':>7} | {'R@100':>7} | {'Time (ms)':>10}")
    print(f"  {'-'*80}")

    for name in sorted(pipelines.keys()):
        p = pipelines[name]
        print(f"  {name:<30} | {p['chamfer_evals']:>13} | {p['r10']:>7.4f} | {p['r100']:>7.4f} | {p['time_ms']:>10.1f}")

    # Find iso-recall matches
    print(f"\n{'='*70}")
    print("KEY EFFICIENCY FINDINGS")
    print(f"{'='*70}")

    # What recall does MUVERA@200 achieve?
    muvera_200 = pipelines.get("[A] MUVERA W=200", {})
    if muvera_200:
        m_r10 = muvera_200["r10"]
        m_r100 = muvera_200["r100"]
        print(f"\n  MUVERA@200:  R@10={m_r10:.4f}, R@100={m_r100:.4f}")

        # Find smallest TCI W that matches or exceeds MUVERA@200 R@10
        for W in [50, 100, 200, 500]:
            tci_key = f"[C] TCI W={W}"
            if tci_key in pipelines:
                t = pipelines[tci_key]
                if t["r10"] >= m_r10:
                    savings = 200 / W
                    print(f"  TCI@{W}:     R@10={t['r10']:.4f}, R@100={t['r100']:.4f}")
                    print(f"  → TCI@{W} matches MUVERA@200 recall with {savings:.1f}x fewer Chamfer evals")
                    break

    # What recall does MUVERA@1000 achieve?
    muvera_1000 = pipelines.get("[A] MUVERA W=1000", {})
    if muvera_1000:
        m_r10 = muvera_1000["r10"]
        print(f"\n  MUVERA@1000: R@10={m_r10:.4f}")

        for W in [50, 100, 200, 500]:
            tci_key = f"[C] TCI W={W}"
            if tci_key in pipelines:
                t = pipelines[tci_key]
                if t["r10"] >= m_r10:
                    savings = 1000 / W
                    print(f"  TCI@{W}:      R@10={t['r10']:.4f}")
                    print(f"  → TCI@{W} matches MUVERA@1000 recall with {savings:.1f}x fewer Chamfer evals")
                    break

    # ================================================================
    # Step 6: Scoring operations table
    # ================================================================
    dim = corpus_tokens[0].shape[1]
    avg_doc_tok = int(avg_tokens)
    avg_q_tok = int(np.mean([len(q) for q in query_tokens]))

    print(f"\n{'='*70}")
    print("SCORING OPERATIONS (implementation-independent)")
    print(f"{'='*70}")
    print(f"  Corpus: {n_docs} docs, avg {avg_doc_tok} tokens, dim={dim}")
    print(f"  Query: avg {avg_q_tok} tokens")

    fde_dim = doc_fdes.shape[1]
    coarse_dim = doc_fdes_coarse.shape[1]

    print(f"\n  {'Pipeline':<45} | {'FDE ops':>12} | {'TCI ops':>10} | {'Chamfer ops':>12} | {'Total ops':>12}")
    print(f"  {'-'*100}")

    configs = [
        ("[A] FDE→Chamfer(200)", n_docs * fde_dim, 0, 200 * avg_q_tok * avg_doc_tok),
        ("[A] FDE→Chamfer(1000)", n_docs * fde_dim, 0, 1000 * avg_q_tok * avg_doc_tok),
        (f"[C] FDE→TCI(200)→Chamfer(200)", n_docs * fde_dim, 1000 * avg_q_tok * K, 200 * avg_q_tok * avg_doc_tok),
        (f"[D] Cascade→FDE→TCI(200)→Chamfer(200)", n_docs * coarse_dim + 5000 * fde_dim, 1000 * avg_q_tok * K, 200 * avg_q_tok * avg_doc_tok),
    ]

    for name, fde_ops, tci_ops, chamfer_ops in configs:
        total = fde_ops + tci_ops + chamfer_ops
        print(f"  {name:<45} | {fde_ops:>12,} | {tci_ops:>10,} | {chamfer_ops:>12,} | {total:>12,}")

    # Savings
    baseline_ops = configs[1][1] + configs[1][2] + configs[1][3]  # MUVERA@1000
    tci_ops_total = configs[2][1] + configs[2][2] + configs[2][3]  # TCI@200
    print(f"\n  MUVERA@1000 total ops: {baseline_ops:,}")
    print(f"  TCI@200 total ops:     {tci_ops_total:,}")
    print(f"  Reduction:             {baseline_ops/tci_ops_total:.1f}x")

    # Save results
    out_dir = f"./results/efficiency_{dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "dataset": dataset_name,
        "n_docs": n_docs,
        "n_queries": n_eval,
        "avg_doc_tokens": avg_doc_tok,
        "avg_query_tokens": avg_q_tok,
        "dim": dim,
        "fde_dim": fde_dim,
        "coarse_dim": coarse_dim,
        "timing_ms_per_query": {
            "fde_full_search": fde_full_time * 1000,
            "fde_cascade_search": cascade_time * 1000,
            "cascade_speedup": fde_full_time / cascade_time,
            "tci_scoring_1000": tci_time * 1000,
            "chamfer_per_doc": {str(W): chamfer_times[W] / W * 1000 for W in chamfer_times},
        },
        "pipelines": pipelines,
        "indexing_time_s": {
            "fde_encoding": fde_encode_time,
            "tci_building": tci_build_time,
        },
    }

    with open(os.path.join(out_dir, "efficiency_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {out_dir}/efficiency_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--n-queries", type=int, default=200,
                        help="Number of queries to benchmark (default 200)")
    args = parser.parse_args()
    run(args)
