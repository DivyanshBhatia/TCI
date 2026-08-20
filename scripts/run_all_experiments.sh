#!/bin/bash
# ============================================================
# MASTER EXPERIMENT RUNNER FOR WSDM 2027 RESUBMISSION
# Addresses all CIKM reviewer concerns
# ============================================================
#
# PREREQUISITES:
#   pip install numpy scikit-learn scipy tqdm torch datasets transformers beir
#
# DIRECTORY STRUCTURE:
#   data/
#     scifact_colbertv2/       # corpus_flat.npy, corpus_lengths.npy, 
#     nfcorpus_colbertv2/      # query_flat.npy, query_lengths.npy, qrels.json
#     fiqa_colbertv2/
#     scidocs_colbertv2/
#     trec_covid_colbertv2/
#     arguana_colbertv2/
#     touche_colbertv2/
#     quora_colbertv2/
#     lotte_sci_colbertv2/
#     msmarco_colbertv2/
#     fever_colbertv2/
#     scifact_xtr/
#     fiqa_xtr/
#     vidore_v3_finance_colpali/
#     vidore_v3_industrial_colpali/
#     vidore_docvqa_colpali/
#     vidore_arxivqa_colpali/
#     vidore_shiftproject_colpali/
#     vidore_tabfquad_colpali/
#     vidore_v3_finance_colqwen2/
#     vidore_docvqa_colqwen2/
#
# ============================================================

set -e  # Exit on error

RESULTS_DIR="results_wsdm"
mkdir -p $RESULTS_DIR

echo "============================================================"
echo "WSDM 2027 EXPERIMENT PIPELINE"
echo "============================================================"
echo ""

# ============================================================
# EXPERIMENT 1: FULL PIPELINE (Brute-force + nDCG + MRR)
# Addresses: R3-W4 (missing MaxSim baseline)
#            R3-W5 (nDCG not tabulated)
#            R1-D5 (bound vs ranking quality)
# ============================================================
echo ">>> EXPERIMENT 1: Full Pipeline with Brute-Force Baseline"
echo "    Addresses: R3-W4, R3-W5, R1-D5"
echo ""

# Small datasets (brute-force feasible)
for DATASET in scifact_colbertv2 nfcorpus_colbertv2 arguana_colbertv2 \
               touche_colbertv2 scidocs_colbertv2 fiqa_colbertv2 \
               trec_covid_colbertv2 quora_colbertv2 \
               scifact_xtr fiqa_xtr \
               vidore_v3_finance_colpali vidore_docvqa_colpali \
               vidore_arxivqa_colpali vidore_shiftproject_colpali \
               vidore_tabfquad_colpali \
               vidore_v3_finance_colqwen2 vidore_docvqa_colqwen2; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_full_pipeline.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/full_pipeline_${DATASET}.log
        mv full_pipeline_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

# Large datasets (skip brute-force)
for DATASET in lotte_sci_colbertv2 msmarco_colbertv2 fever_colbertv2 \
               vidore_v3_industrial_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running (no brute-force): $DATASET"
        python run_full_pipeline.py --embeddings-dir data/$DATASET \
            --skip-bruteforce \
            2>&1 | tee $RESULTS_DIR/full_pipeline_${DATASET}.log
        mv full_pipeline_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 2: MATCHED-MEMORY COMPARISON
# Addresses: R2 ("How does MUVERA behave with same memory budget?")
#            Meta-review ("accuracy-latency-memory trade-off")
# ============================================================
echo ">>> EXPERIMENT 2: Matched-Memory MUVERA vs TCI"
echo "    Addresses: R2, Meta-review"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 fiqa_colbertv2 \
               scidocs_colbertv2 arguana_colbertv2 \
               vidore_v3_finance_colpali vidore_docvqa_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_matched_memory.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/matched_memory_${DATASET}.log
        mv matched_memory_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 3: K ABLATION ON TEXT DATASETS
# Addresses: R3-W1 ("ablation focuses on two visual datasets")
# ============================================================
echo ">>> EXPERIMENT 3: K Ablation on Text Datasets"
echo "    Addresses: R3-W1"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 fiqa_colbertv2 \
               scidocs_colbertv2 arguana_colbertv2; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_k_ablation_text.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/k_ablation_${DATASET}.log
        mv k_ablation_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 4: COMPRESSION BASELINES (FULL PIPELINE METRICS)
# Addresses: R3-W5 ("results not fully reported in table")
#            R1-D1 ("what's new vs prior compression methods")
# ============================================================
echo ">>> EXPERIMENT 4: Compression Baselines with Full Metrics"
echo "    Addresses: R3-W5, R1-D1"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 fiqa_colbertv2 \
               arguana_colbertv2 \
               vidore_v3_finance_colpali vidore_docvqa_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_compression_pipeline.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/compression_${DATASET}.log
        mv compression_pipeline_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 5: SIGNIFICANCE ON ALL DATASETS (including small)
# Addresses: R3-W2 ("report tests for all datasets or justify")
# ============================================================
echo ">>> EXPERIMENT 5: Significance Tests (ALL datasets)"
echo "    Addresses: R3-W2"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 fiqa_colbertv2 \
               scidocs_colbertv2 trec_covid_colbertv2 arguana_colbertv2 \
               touche_colbertv2 quora_colbertv2 lotte_sci_colbertv2 \
               msmarco_colbertv2 fever_colbertv2 \
               scifact_xtr fiqa_xtr \
               vidore_v3_finance_colpali vidore_v3_industrial_colpali \
               vidore_docvqa_colpali vidore_arxivqa_colpali \
               vidore_shiftproject_colpali vidore_tabfquad_colpali \
               vidore_v3_finance_colqwen2 vidore_docvqa_colqwen2; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_significance.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/significance_${DATASET}.log
        mv significance_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 6: EXTENDED BASELINES (PLAID-1024, Enhanced FDE)
# Addresses: R2 ("not compared against PLAID")
#            Meta-review ("omits comparisons with important baselines")
# ============================================================
echo ">>> EXPERIMENT 6: PLAID-1024 + Enhanced FDE"
echo "    Addresses: R2, Meta-review"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 fiqa_colbertv2 \
               arguana_colbertv2 scidocs_colbertv2 \
               vidore_v3_finance_colpali vidore_v3_industrial_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_extended_baselines.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/extended_${DATASET}.log
        mv extended_baselines_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 7: PLAID-SOURCED CANDIDATES + TCI
# Addresses: R2 ("unclear if TCI provides additive benefit in PLAID")
# ============================================================
echo ">>> EXPERIMENT 7: PLAID + TCI Integration"
echo "    Addresses: R2"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 fiqa_colbertv2 \
               arguana_colbertv2 \
               vidore_v3_finance_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_plaid_tci.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/plaid_tci_${DATASET}.log
        mv plaid_tci_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 8: E/M MARGIN ANALYSIS (ALL DATASETS)
# Addresses: R1-D3 (threshold validation)
#            R1-D4 (what does E/M computation require)
# ============================================================
echo ">>> EXPERIMENT 8: E/M Ratio Computation"
echo "    Addresses: R1-D3, R1-D4"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 fiqa_colbertv2 \
               scidocs_colbertv2 trec_covid_colbertv2 arguana_colbertv2 \
               touche_colbertv2 quora_colbertv2 lotte_sci_colbertv2 \
               vidore_v3_finance_colpali vidore_v3_industrial_colpali \
               vidore_docvqa_colpali vidore_arxivqa_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_margin_analysis.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/margin_${DATASET}.log
        mv margin_analysis_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 9: GPU LATENCY BENCHMARK
# Addresses: R3-W9 ("GPU claim unverified")
# ============================================================
echo ">>> EXPERIMENT 9: GPU Latency Benchmark"
echo "    Addresses: R3 Q4"
echo ""

python benchmark_tci_gpu.py --K 32 2>&1 | tee $RESULTS_DIR/gpu_benchmark_K32.log
python benchmark_tci_gpu.py --K 64 2>&1 | tee $RESULTS_DIR/gpu_benchmark_K64.log

echo ""

# ============================================================
# EXPERIMENT 10: THEORY VALIDATION
# Addresses: R1-D5 (bound vs ranking)
# ============================================================
echo ">>> EXPERIMENT 10: Theorem 5.2 Validation"
echo "    Addresses: R1-D5"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 \
               vidore_v3_finance_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_theory_validation.py --embeddings-dir data/$DATASET \
            2>&1 | tee $RESULTS_DIR/theory_${DATASET}.log
        mv theory_validation_${DATASET}.json $RESULTS_DIR/ 2>/dev/null || true
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""

# ============================================================
# EXPERIMENT 11: ROBUSTNESS (K-MEANS SEED VARIANCE)
# ============================================================
echo ">>> EXPERIMENT 11: K-Means Robustness"
echo ""

for DATASET in scifact_colbertv2 nfcorpus_colbertv2 touche_colbertv2 \
               vidore_v3_finance_colpali; do
    if [ -d "data/$DATASET" ]; then
        echo "  Running: $DATASET"
        python run_kmeans_robustness.py --embeddings-dir data/$DATASET --n-seeds 10 \
            2>&1 | tee $RESULTS_DIR/robustness_${DATASET}.log
    else
        echo "  SKIP: data/$DATASET not found"
    fi
done

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Results saved to: $RESULTS_DIR/"
echo "============================================================"

# ============================================================
# SUMMARY: What goes where in the paper
# ============================================================
cat << 'SUMMARY'

PAPER TABLE MAPPING:
  Table 1: Scoring error        ← run_margin_analysis.py (already have)
  Table 2: E/M predictions      ← run_margin_analysis.py (all datasets)
  Table 3: Main results         ← run_full_pipeline.py (R@10, R@100, nDCG@10, MRR + brute-force row)
  Table 4: Significance         ← run_full_pipeline.py (ALL datasets, not just "adequate")
  Table 5: Inversion baselines  ← run_extended_baselines.py + run_compression_baselines.py
  Table 6: Iso-recall           ← run_efficiency.py (already have)
  Table 7: Latency              ← benchmark_tci_gpu.py (add GPU column)
  Table 8: K ablation           ← run_k_ablation_text.py (text) + existing visual
  Table 9: Cross-model          ← run_full_pipeline.py (ColQwen2 datasets)
  
  NEW TABLE: Matched-memory     ← run_matched_memory.py
  NEW TABLE: Compression full   ← run_compression_pipeline.py (R@10, R@100, nDCG, not just inversion)
  NEW TABLE: PLAID integration  ← run_plaid_tci.py

SUMMARY
