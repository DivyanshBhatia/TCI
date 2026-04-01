# ============================================================
# Encode ViDoRe V3 Industrial with ColPali
# 5,244 pages, 1,698 queries — industrial documents
# ============================================================
# !pip install colpali-engine==0.3.4 transformers==4.46.3 peft==0.11.1
# !pip install torch datasets pillow

import os, json, numpy as np
from datasets import load_dataset
from tqdm import tqdm
import torch
from PIL import Image

# ============================================================
# CELL 1: Load dataset
# ============================================================
print("Loading ViDoRe V3 Industrial...")

corpus_ds = load_dataset("vidore/vidore_v3_industrial", "corpus", split="test")
print(f"Corpus: {len(corpus_ds)} pages")

queries_ds = load_dataset("vidore/vidore_v3_industrial", "queries", split="test")
print(f"Queries: {len(queries_ds)} queries")

qrels_ds = load_dataset("vidore/vidore_v3_industrial", "qrels", split="test")
print(f"Qrels: {len(qrels_ds)} judgments")

# ============================================================
# CELL 2: Build mappings
# ============================================================
corpus_ids = []
corpus_images = []
for row in corpus_ds:
    corpus_ids.append(str(row["corpus_id"]))
    corpus_images.append(row["image"])

cid_to_idx = {cid: i for i, cid in enumerate(corpus_ids)}
print(f"Corpus pages: {len(corpus_ids)}")

query_ids = []
query_texts = []
for row in queries_ds:
    query_ids.append(str(row["query_id"]))
    query_texts.append(row["query"])

qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
print(f"Queries: {len(query_ids)}")

qrels = {}
for row in qrels_ds:
    qid = str(row["query_id"])
    cid = str(row["corpus_id"])
    score = int(row["score"])
    if qid not in qrels:
        qrels[qid] = {}
    qrels[qid][cid] = score

print(f"Qrels: {len(qrels)} queries with judgments")

ground_truth = []
for qid in query_ids:
    if qid in qrels:
        rel = [cid_to_idx[cid] for cid, s in qrels[qid].items()
               if s > 0 and cid in cid_to_idx]
        ground_truth.append(rel)
    else:
        ground_truth.append([])

n_with_gt = sum(1 for gt in ground_truth if len(gt) > 0)
print(f"Queries with ground truth: {n_with_gt}/{len(query_ids)}")
avg_rel = np.mean([len(gt) for gt in ground_truth if len(gt) > 0])
print(f"Avg relevant pages/query: {avg_rel:.1f}")

# ============================================================
# CELL 3: Load ColPali model
# ============================================================
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDevice: {device}")

model_name = "vidore/colpali-v1.3"
print(f"Loading {model_name}...")

model = ColPali.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
)
processor = ColPaliProcessor.from_pretrained(model_name)
model.eval()
print("Model loaded!")

# ============================================================
# CELL 3.5: Monkey-patch the image token count mismatch
# ============================================================
# Known bug: processor creates 1025 <image> tokens but vision
# encoder outputs 1024 (32x32 patches from 448x448 / 14).
# Fix: after processing, trim input_ids to match 1024 image tokens.

_IMAGE_TOKEN_ID = processor.tokenizer.convert_tokens_to_ids("<image>")
print(f"Image token ID: {_IMAGE_TOKEN_ID}")

def fix_image_token_count(batch, expected_image_tokens=1024):
    """Trim extra <image> tokens from input_ids to match vision encoder output."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    fixed_ids = []
    fixed_mask = []
    for i in range(input_ids.shape[0]):
        ids = input_ids[i]
        mask = attention_mask[i]

        # Count image tokens
        image_mask = (ids == _IMAGE_TOKEN_ID)
        n_image = image_mask.sum().item()

        if n_image > expected_image_tokens:
            # Find positions of image tokens
            image_positions = torch.where(image_mask)[0]
            # Remove the extra ones from the end of the image token block
            n_remove = n_image - expected_image_tokens
            remove_positions = image_positions[-n_remove:]

            keep_mask = torch.ones_like(ids, dtype=torch.bool)
            keep_mask[remove_positions] = False

            ids = ids[keep_mask]
            mask = mask[keep_mask]

        fixed_ids.append(ids)
        fixed_mask.append(mask)

    # Pad to same length
    max_len = max(len(x) for x in fixed_ids)
    padded_ids = torch.zeros(len(fixed_ids), max_len, dtype=input_ids.dtype, device=input_ids.device)
    padded_mask = torch.zeros(len(fixed_ids), max_len, dtype=attention_mask.dtype, device=attention_mask.device)

    for i in range(len(fixed_ids)):
        padded_ids[i, :len(fixed_ids[i])] = fixed_ids[i]
        padded_mask[i, :len(fixed_mask[i])] = fixed_mask[i]

    batch["input_ids"] = padded_ids
    batch["attention_mask"] = padded_mask
    return batch

print("Image token fix applied!")

# ============================================================
# CELL 4: Encode corpus pages
# ============================================================
print(f"\nEncoding {len(corpus_images)} pages...")

all_doc_embs = []
all_doc_lengths = []
batch_size = 4
n_errors = 0

for i in tqdm(range(0, len(corpus_images), batch_size), desc="Encoding pages"):
    batch_images = corpus_images[i:i+batch_size]

    try:
        batch = processor.process_images(batch_images).to(device)
        batch = fix_image_token_count(batch)

        with torch.no_grad():
            embeddings = model(**batch)

        for emb in embeddings:
            emb_np = emb.cpu().float().numpy()
            all_doc_embs.append(emb_np)
            all_doc_lengths.append(len(emb_np))

    except Exception as e:
        # Fallback: encode one by one
        for j, img in enumerate(batch_images):
            try:
                single_batch = processor.process_images([img]).to(device)
                single_batch = fix_image_token_count(single_batch)
                with torch.no_grad():
                    emb = model(**single_batch)[0]
                emb_np = emb.cpu().float().numpy()
                all_doc_embs.append(emb_np)
                all_doc_lengths.append(len(emb_np))
            except Exception as e2:
                print(f"  Warning: page {i+j} failed ({e2}), skipping")
                # Use zeros as placeholder to maintain alignment
                dim = all_doc_embs[0].shape[1] if all_doc_embs else 128
                n_patches = all_doc_lengths[0] if all_doc_lengths else 1024
                all_doc_embs.append(np.zeros((n_patches, dim), dtype=np.float32))
                all_doc_lengths.append(n_patches)
                n_errors += 1

print(f"Encoded {len(all_doc_embs)} pages ({n_errors} failures)")
print(f"Avg patches/page: {np.mean(all_doc_lengths):.0f}")

# ============================================================
# CELL 5: Encode queries
# ============================================================
print(f"\nEncoding {len(query_texts)} queries...")

all_query_embs = []
all_query_lengths = []
batch_size_q = 16

for i in tqdm(range(0, len(query_texts), batch_size_q), desc="Encoding queries"):
    batch_texts = query_texts[i:i+batch_size_q]
    batch = processor.process_queries(batch_texts).to(device)

    with torch.no_grad():
        embeddings = model(**batch)

    for emb in embeddings:
        emb_np = emb.cpu().float().numpy()
        all_query_embs.append(emb_np)
        all_query_lengths.append(len(emb_np))

print(f"Encoded {len(all_query_embs)} queries")
print(f"Avg tokens/query: {np.mean(all_query_lengths):.0f}")

# ============================================================
# CELL 6: Save
# ============================================================
out_dir = "/content/vidore_v3_industrial_colpali"
os.makedirs(out_dir, exist_ok=True)

corpus_flat = np.vstack(all_doc_embs).astype(np.float32)
query_flat = np.vstack(all_query_embs).astype(np.float32)

np.save(f"{out_dir}/corpus_flat.npy", corpus_flat)
np.save(f"{out_dir}/corpus_lengths.npy", np.array(all_doc_lengths, dtype=np.int32))
np.save(f"{out_dir}/query_flat.npy", query_flat)
np.save(f"{out_dir}/query_lengths.npy", np.array(all_query_lengths, dtype=np.int32))

with open(f"{out_dir}/qrels.json", 'w') as f:
    json.dump({'ground_truth': ground_truth}, f)

print(f"\nSaved to {out_dir}")
print(f"  Corpus: {corpus_flat.shape}")
print(f"  Queries: {query_flat.shape}")
print(f"  Avg patches/page: {np.mean(all_doc_lengths):.0f}")
print(f"  Embedding dim: {corpus_flat.shape[1]}")
print(f"  Queries with GT: {n_with_gt}/{len(query_ids)}")
if n_errors > 0:
    print(f"  WARNING: {n_errors} pages failed encoding")

import shutil
shutil.make_archive(out_dir, 'zip', '/content', 'vidore_v3_industrial_colpali')
print(f"  Zipped: {out_dir}.zip")

print(f"\nRun locally:")
print(f"  python run_significance.py --embeddings-dir vidore_v3_industrial_colpali")
print(f"  python run_margin_analysis.py --embeddings-dir vidore_v3_industrial_colpali")
