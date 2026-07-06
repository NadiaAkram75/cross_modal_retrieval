import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

DATA_ROOT = r"data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

def clean_ids(ids):
    return [str(x[0]) if isinstance(x, (list, np.ndarray)) else str(x) for x in ids]

def cosine_distance_matrix(x, query):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    query = query / (np.linalg.norm(query) + 1e-8)
    return 1 - np.dot(x, query)

def load_slice(case_id):
    path = os.path.join(DATA_ROOT, case_id, f"{case_id}_flair.nii")
    img = nib.load(path).get_fdata()
    return img[:, :, img.shape[2] // 2]

parser = argparse.ArgumentParser()
parser.add_argument("--query-index", type=int, default=0)
parser.add_argument("--top-k", type=int, default=5)
parser.add_argument("--semantic-weight", type=float, default=0.75)
args = parser.parse_args()

embeddings = np.load("embeddings.npy")
case_ids = clean_ids(np.load("case_ids.npy", allow_pickle=True))
semantic_features = np.load("semantic_features.npy")
semantic_case_ids = clean_ids(np.load("semantic_case_ids.npy", allow_pickle=True))

visual_map = {c: embeddings[i] for i, c in enumerate(case_ids)}
semantic_map = {c: semantic_features[i] for i, c in enumerate(semantic_case_ids)}
common_ids = [c for c in case_ids if c in semantic_map]

visual = np.array([visual_map[c] for c in common_ids])
semantic = np.array([semantic_map[c] for c in common_ids])
semantic = (semantic - semantic.mean(axis=0)) / (semantic.std(axis=0) + 1e-8)

query_id = common_ids[args.query_index]
visual_dist = cosine_distance_matrix(visual, visual[args.query_index])
semantic_dist = cosine_distance_matrix(semantic, semantic[args.query_index])
combined_dist = args.semantic_weight * semantic_dist + (1 - args.semantic_weight) * visual_dist

ranking = [i for i in np.argsort(combined_dist) if common_ids[i] != query_id]
top_indices = ranking[:args.top_k]
show_ids = [query_id] + [common_ids[i] for i in top_indices]

plt.figure(figsize=(15, 3))

for i, case_id in enumerate(show_ids):
    plt.subplot(1, len(show_ids), i + 1)
    plt.imshow(load_slice(case_id), cmap="gray")
    title = "Query\n" + case_id if i == 0 else "Retrieved\n" + case_id
    plt.title(title, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig("semantic_retrieval_result.png", dpi=200)
print("Saved visualization: semantic_retrieval_result.png")