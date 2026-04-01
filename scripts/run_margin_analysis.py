"""
E/M Ratio Computation (Margin Analysis).
Computes the error-to-margin ratio ρ for a dataset, predicting whether TCI will help.

Usage:
  python run_margin_analysis.py --embeddings-dir data/scifact_colbertv2
  
Output: ρ > 1.3 → TCI helps, ρ < 1.0 → TCI unnecessary.
"""

import argparse
import json
import os
import numpy as np
from sklearn.cluster import KMeans
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

    return corpus_tokens, query_tokens, qrels["ground_truth"], os.path.basename(embeddings_dir.rstrip("/")), corpus_lengths


def chamfer_score(q_tokens, d_tokens):
    sim = q_tokens @ d_tokens.T
    return float(sim.max(axis=1).sum())


def encode_fde(doc_tokens, R=10, seed=42):
    rng = np.random.RandomState(seed)
    dim = doc_tokens.shape[1]
    fde = np.zeros(R * 2 * dim, dtype=np.float32)
    n_tokens = len(doc_tokens)
    for r in range(R):
        assignments = rng.randint(0, 2, size=n_tokens)
        for b in range(2):
            mask = assignments == b
            if mask.any():
                start = (r * 2 + b) * dim
                fde[start:start+dim] = doc_tokens[mask].mean(axis=0)
    return fde


def tci_score(q_tokens, centroids):
    sim = q_tokens @ centroids.T
    return float(sim.max(axis=1).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", required=True)
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--K", type=int, default=32)
    args = parser.parse_args()

    corpus_tokens, query_tokens, ground_truth, name, corpus_lengths = load_embeddings(args.embeddings_dir)
    n_docs = len(corpus_tokens)

    print("=" * 70)
    print(f"E/M RATIO ANALYSIS: {name}")
    print("=" * 70)
    print(f"  Docs: {n_docs}, Queries: {len(query_tokens)}, Avg |D|: {np.mean(corpus_lengths):.1f}")

    # Sample queries with ground truth
    valid = [i for i in range(len(query_tokens)) if len(ground_truth[i]) > 0]
    rng = np.random.RandomState(42)
    if len(valid) > args.n_queries:
        valid = sorted(rng.choice(valid, size=args.n_queries, replace=False).tolist())

    # Encode FDEs
    print("Encoding FDEs...")
    doc_fdes = np.array([encode_fde(d) for d in corpus_tokens])

    # Build TCI index
    print(f"Building TCI-{args.K} index...")
    tci_centroids = []
    for tokens in corpus_tokens:
        k = min(args.K, len(tokens))
        if k < 2:
            tci_centroids.append(tokens)
            continue
        km = KMeans(n_clusters=k, n_init=1, max_iter=50, random_state=42)
        km.fit(tokens)
        tci_centroids.append(km.cluster_centers_.astype(np.float32))

    # Compute margins and errors
    fde_errors, tci_errors, margins = [], [], []
    n_pairs, fde_inv, tci_inv = 0, 0, 0

    for qi in tqdm(valid, desc="Analyzing"):
        q_tok = query_tokens[qi]
        gt_set = set(ground_truth[qi])

        # Get FDE top candidates
        q_fde = encode_fde(q_tok)
        fde_scores = doc_fdes @ q_fde
        top_indices = np.argsort(-fde_scores)[:1000]

        rel_in_top = [di for di in top_indices if di in gt_set]
        nonrel_in_top = [di for di in top_indices if di not in gt_set]

        if not rel_in_top or not nonrel_in_top:
            continue

        for di_rel in rel_in_top[:5]:
            chamfer_rel = chamfer_score(q_tok, corpus_tokens[di_rel])
            fde_rel = float(fde_scores[di_rel])
            tci_rel = tci_score(q_tok, tci_centroids[di_rel])

            # FDE error on relevant doc
            if abs(chamfer_rel) > 1e-8:
                fde_errors.append(abs(fde_rel - chamfer_rel) / abs(chamfer_rel))
                tci_errors.append(abs(tci_rel - chamfer_rel) / abs(chamfer_rel))

            # Hardest negative
            best_neg_chamfer = -1e9
            best_neg_idx = -1
            for di_neg in nonrel_in_top[:20]:
                c = chamfer_score(q_tok, corpus_tokens[di_neg])
                if c > best_neg_chamfer:
                    best_neg_chamfer = c
                    best_neg_idx = di_neg

            if best_neg_idx >= 0 and chamfer_rel > best_neg_chamfer:
                margin = (chamfer_rel - best_neg_chamfer) / chamfer_rel
                margins.append(margin)

                n_pairs += 1
                if fde_rel < float(fde_scores[best_neg_idx]):
                    fde_inv += 1
                if tci_rel < tci_score(q_tok, tci_centroids[best_neg_idx]):
                    tci_inv += 1

    fde_err = np.mean(fde_errors)
    tci_err = np.mean(tci_errors)
    med_margin = np.median(margins)
    rho = fde_err / med_margin if med_margin > 0 else float('inf')

    prediction = "TCI_HELPS" if rho > 1.3 else ("TRANSITIONAL" if rho > 1.0 else "NO_HELP")

    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"  FDE scoring error (mean):  {fde_err:.4f}")
    print(f"  TCI scoring error (mean):  {tci_err:.4f}")
    print(f"  Score margin (median):     {med_margin:.4f}")
    print(f"  E/M ratio (ρ):             {rho:.2f}")
    print(f"  Prediction:                {prediction}")
    print(f"  FDE inversion rate:        {fde_inv/max(n_pairs,1):.4f}")
    print(f"  TCI inversion rate:        {tci_inv/max(n_pairs,1):.4f}")
    print(f"  Inversion reduction:       {(1 - tci_inv/max(fde_inv,1))*100:.1f}%")

    results = {
        "dataset": name,
        "n_docs": n_docs,
        "n_queries": len(query_tokens),
        "avg_doc_tokens": float(np.mean(corpus_lengths)),
        "n_pairs_analyzed": n_pairs,
        "margin": {"mean": float(np.mean(margins)), "median": float(med_margin)},
        "scoring_error": {
            "fde_relevant": float(fde_err),
            "tci_relevant": float(tci_err),
        },
        "inversions": {
            "fde_rate": fde_inv / max(n_pairs, 1),
            "tci_rate": tci_inv / max(n_pairs, 1),
            "fde_count": fde_inv,
            "tci_count": tci_inv,
            "reduction_pct": (1 - tci_inv / max(fde_inv, 1)) * 100,
        },
        "em_ratio": float(rho),
        "prediction": prediction,
    }

    out_file = f"margin_analysis_{name}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
