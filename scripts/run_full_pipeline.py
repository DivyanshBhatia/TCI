"""
Full Pipeline with Brute-Force MaxSim Baseline.
Addresses R3-W4 (missing ColBERT baseline) and R3-W5 (missing nDCG table).

Runs three pipelines on every dataset:
  1. Brute-force MaxSim (exact scoring, upper bound)
  2. MUVERA: FDE → Chamfer rerank
  3. TCI: FDE → TCI rescore → Chamfer rerank

Reports: R@10, R@100, nDCG@10, MRR for all three.

Usage:
  python run_full_pipeline.py --embeddings-dir data/scifact_colbertv2
  python run_full_pipeline.py --embeddings-dir data/nfcorpus_colbertv2
  # Run on ALL datasets for the complete table
"""

import argparse
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon
from tqdm import tqdm


def load_embeddings(embeddings_dir):
    corpus_flat = np.load(os.path.join(embeddings_dir, "corpus_flat.npy"))
    corpus_lengths = np.load(os.path.join(embeddings_dir, "corpus_lengths.npy"))
    query_flat = np.load(os.path.join(embeddings_dir, "query_flat.npy"))
    query_lengths = np.load(os.path.join(embeddings_dir, "query_lengths.npy"))
    with open(os.path.join(embeddings_dir, "qrels.json")) as f:
        qrels = json.load(f)

    corpus_tokens, offset = [], 0
    for l in corpus_lengths:
        corpus_tokens.append(corpus_flat[offset:offset+l])
        offset += l

    query_tokens, offset = [], 0
    for l in query_lengths:
        query_tokens.append(query_flat[offset:offset+l])
        offset += l

    return corpus_tokens, query_tokens, qrels["ground_truth"], \
           os.path.basename(embeddings_dir.rstrip("/"))


def chamfer_score(q, d):
    return float((q @ d.T).max(axis=1).sum())


def encode_fde(doc_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = doc_tokens.shape[1]
    fde = np.zeros(R * 2 * dim, dtype=np.float32)
    for r in range(R):
        assignments = rng.randint(0, 2, size=len(doc_tokens))
        for b in range(2):
            mask = assignments == b
            if mask.any():
                start = (r * 2 + b) * dim
                fde[start:start+dim] = doc_tokens[mask].mean(axis=0)
    return fde


def encode_fde_query(q_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = q_tokens.shape[1]
    fde = np.zeros(R * 2 * dim, dtype=np.float32)
    for r in range(R):
        assignments = rng.randint(0, 2, size=len(q_tokens))
        for b in range(2):
            mask = assignments == b
            if mask.any():
                start = (r * 2 + b) * dim
                fde[start:start+dim] = q_tokens[mask].sum(axis=0) / R
    return fde


def compute_ndcg(ranking, ground_truth, k=10):
    """Compute nDCG@k."""
    gt_set = set(ground_truth)
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k]):
        if doc_id in gt_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
    # Ideal DCG
    n_rel = min(len(gt_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(ranking, ground_truth):
    """Compute MRR."""
    gt_set = set(ground_truth)
    for i, doc_id in enumerate(ranking):
        if doc_id in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def run_pipeline(query_tokens, corpus_tokens, ground_truth,
                 doc_fdes, tci_centroids,
                 W_prime=1000, W=200, K=32,
                 run_bruteforce=True):
    """Run all three pipelines and compute all metrics."""

    n_queries = len(query_tokens)
    n_docs = len(corpus_tokens)

    # Per-query metrics storage
    metrics = {
        'bruteforce': {'r10': [], 'r100': [], 'ndcg10': [], 'mrr': []},
        'muvera': {'r10': [], 'r100': [], 'ndcg10': [], 'mrr': []},
        'tci': {'r10': [], 'r100': [], 'ndcg10': [], 'mrr': []},
    }

    for qi in tqdm(range(n_queries), desc="Queries"):
        gt = ground_truth[qi]
        if not gt:
            continue

        q_tok = query_tokens[qi]
        gt_set = set(gt)
        n_rel = len(gt_set)

        # ============================================
        # Pipeline 1: Brute-force MaxSim (exact)
        # ============================================
        if run_bruteforce:
            bf_scores = []
            for di in range(n_docs):
                bf_scores.append((di, chamfer_score(q_tok, corpus_tokens[di])))
            bf_scores.sort(key=lambda x: -x[1])
            bf_ranking = [di for di, _ in bf_scores]

            for metric_name, metric_fn, k_val in [
                ('r10', lambda r, g, k: len(set(r[:k]) & g) / len(g), 10),
                ('r100', lambda r, g, k: len(set(r[:k]) & g) / len(g), 100),
                ('ndcg10', lambda r, g, k: compute_ndcg(r, g, k), 10),
                ('mrr', lambda r, g, k: compute_mrr(r, g), None),
            ]:
                if k_val is not None:
                    metrics['bruteforce'][metric_name].append(metric_fn(bf_ranking, gt, k_val))
                else:
                    metrics['bruteforce'][metric_name].append(metric_fn(bf_ranking, gt, None))

        # ============================================
        # FDE candidate retrieval (shared by MUVERA and TCI)
        # ============================================
        q_fde = encode_fde_query(q_tok)
        fde_scores = doc_fdes @ q_fde
        top_W_prime = np.argsort(-fde_scores)[:W_prime]

        # ============================================
        # Pipeline 2: MUVERA (FDE → Chamfer top-W)
        # ============================================
        muvera_candidates = top_W_prime[:W]
        muvera_chamfer = [(di, chamfer_score(q_tok, corpus_tokens[di]))
                          for di in muvera_candidates]
        muvera_chamfer.sort(key=lambda x: -x[1])
        muvera_ranking = [di for di, _ in muvera_chamfer]

        metrics['muvera']['r10'].append(len(set(muvera_ranking[:10]) & gt_set) / n_rel)
        metrics['muvera']['r100'].append(len(set(muvera_ranking[:100]) & gt_set) / n_rel)
        metrics['muvera']['ndcg10'].append(compute_ndcg(muvera_ranking, gt))
        metrics['muvera']['mrr'].append(compute_mrr(muvera_ranking, gt))

        # ============================================
        # Pipeline 3: TCI (FDE → TCI → Chamfer top-W)
        # ============================================
        tci_scores_list = [(di, float((q_tok @ tci_centroids[di].T).max(axis=1).sum()))
                           for di in top_W_prime]
        tci_scores_list.sort(key=lambda x: -x[1])
        tci_candidates = [di for di, _ in tci_scores_list[:W]]

        tci_chamfer = [(di, chamfer_score(q_tok, corpus_tokens[di]))
                       for di in tci_candidates]
        tci_chamfer.sort(key=lambda x: -x[1])
        tci_ranking = [di for di, _ in tci_chamfer]

        metrics['tci']['r10'].append(len(set(tci_ranking[:10]) & gt_set) / n_rel)
        metrics['tci']['r100'].append(len(set(tci_ranking[:100]) & gt_set) / n_rel)
        metrics['tci']['ndcg10'].append(compute_ndcg(tci_ranking, gt))
        metrics['tci']['mrr'].append(compute_mrr(tci_ranking, gt))

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--W-prime", type=int, default=1000)
    parser.add_argument("--W", type=int, default=200)
    parser.add_argument("--skip-bruteforce", action="store_true",
                        help="Skip brute-force (slow on large corpora)")
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name = load_embeddings(args.embeddings_dir)

    print("=" * 70)
    print(f"FULL PIPELINE: {name}")
    print(f"  Docs: {len(corpus_tokens)}, Queries: {len(query_tokens)}")
    print(f"  K={args.K}, W'={args.W_prime}, W={args.W}")
    print("=" * 70)

    # Encode FDEs
    print("\nEncoding FDEs...")
    doc_fdes = np.array([encode_fde(d) for d in tqdm(corpus_tokens, desc="FDE")])

    # Build TCI index
    print(f"\nBuilding TCI-{args.K} index...")
    tci_centroids = []
    for tokens in tqdm(corpus_tokens, desc="TCI"):
        k = min(args.K, len(tokens))
        if k < 2:
            tci_centroids.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        tci_centroids.append(km.cluster_centers_.astype(np.float32))

    # Run pipelines
    print("\nRunning pipelines...")
    metrics = run_pipeline(
        query_tokens, corpus_tokens, ground_truth,
        doc_fdes, tci_centroids,
        W_prime=args.W_prime, W=args.W, K=args.K,
        run_bruteforce=not args.skip_bruteforce
    )

    # Compute aggregates
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Pipeline':<15} {'R@10':>8} {'R@100':>8} {'nDCG@10':>8} {'MRR':>8}")
    print("-" * 55)

    results = {"dataset": name, "n_docs": len(corpus_tokens),
               "n_queries": len(query_tokens), "K": args.K}

    for pipeline in ['bruteforce', 'muvera', 'tci']:
        m = metrics[pipeline]
        if not m['r10']:
            continue
        r10 = np.mean(m['r10'])
        r100 = np.mean(m['r100'])
        ndcg = np.mean(m['ndcg10'])
        mrr = np.mean(m['mrr'])
        label = {'bruteforce': 'Brute-force', 'muvera': 'MUVERA@200',
                 'tci': f'TCI-{args.K}@200'}[pipeline]
        print(f"  {label:<13} {r10:>8.4f} {r100:>8.4f} {ndcg:>8.4f} {mrr:>8.4f}")

        results[pipeline] = {
            'R@10': float(r10), 'R@100': float(r100),
            'nDCG@10': float(ndcg), 'MRR': float(mrr),
            'per_query_r100': [float(x) for x in m['r100']],
            'per_query_ndcg10': [float(x) for x in m['ndcg10']],
        }

    # Significance tests (TCI vs MUVERA, all queries including small sets)
    if metrics['tci']['r100'] and metrics['muvera']['r100']:
        tci_r100 = np.array(metrics['tci']['r100'])
        muv_r100 = np.array(metrics['muvera']['r100'])
        diff = tci_r100 - muv_r100
        nonzero = diff[diff != 0]

        n_improved = int(np.sum(diff > 0))
        n_hurt = int(np.sum(diff < 0))
        n_tied = int(np.sum(diff == 0))

        print(f"\n  Significance (R@100): Win={n_improved}, Loss={n_hurt}, Tie={n_tied}")

        if len(nonzero) >= 10:
            stat, pval = wilcoxon(nonzero)
            print(f"  Wilcoxon p-value: {pval:.2e}")
            results['significance'] = {
                'wilcoxon_p': float(pval), 'wilcoxon_stat': float(stat),
                'n_improved': n_improved, 'n_hurt': n_hurt, 'n_tied': n_tied,
                'n_queries': len(diff),
            }
        else:
            print(f"  Too few non-tied pairs ({len(nonzero)}) for Wilcoxon test")
            results['significance'] = {
                'n_improved': n_improved, 'n_hurt': n_hurt, 'n_tied': n_tied,
                'n_queries': len(diff), 'note': 'insufficient non-tied pairs',
            }

    # Recovery percentage (how much of brute-force gap does TCI close?)
    if 'bruteforce' in results and 'muvera' in results and 'tci' in results:
        for metric in ['R@10', 'R@100', 'nDCG@10', 'MRR']:
            bf = results['bruteforce'][metric]
            muv = results['muvera'][metric]
            tci = results['tci'][metric]
            gap = bf - muv
            if gap > 0.001:
                recovery = (tci - muv) / gap * 100
                print(f"  {metric} recovery: {recovery:.1f}% of brute-force gap")

    out_file = f"full_pipeline_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
