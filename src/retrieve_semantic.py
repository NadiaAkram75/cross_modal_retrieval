import argparse
import numpy as np

def clean_ids(ids):
    return [str(x[0]) if isinstance(x, (list, np.ndarray)) else str(x) for x in ids]

def cosine_distance_matrix(x, query):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    query = query / (np.linalg.norm(query) + 1e-8)
    return 1 - np.dot(x, query)

parser = argparse.ArgumentParser()
parser.add_argument("--query-index", type=int, default=0)
parser.add_argument("--top-k", type=int, default=5)
parser.add_argument("--semantic-weight", type=float, default=0.75)
args = parser.parse_args()

embeddings = np.load("embeddings.npy")
case_ids = clean_ids(np.load("case_ids.npy", allow_pickle=True))

semantic_features = np.load("semantic_features.npy")
semantic_case_ids = clean_ids(np.load("semantic_case_ids.npy", allow_pickle=True))

visual_map = {case_id: embeddings[i] for i, case_id in enumerate(case_ids)}
semantic_map = {case_id: semantic_features[i] for i, case_id in enumerate(semantic_case_ids)}

common_ids = [case_id for case_id in case_ids if case_id in semantic_map]

visual = np.array([visual_map[c] for c in common_ids])
semantic = np.array([semantic_map[c] for c in common_ids])

semantic = (semantic - semantic.mean(axis=0)) / (semantic.std(axis=0) + 1e-8)

query_id = common_ids[args.query_index]
query_visual = visual[args.query_index]
query_semantic = semantic[args.query_index]

visual_dist = cosine_distance_matrix(visual, query_visual)
semantic_dist = cosine_distance_matrix(semantic, query_semantic)

combined_dist = (
    args.semantic_weight * semantic_dist
    + (1 - args.semantic_weight) * visual_dist
)

ranking = np.argsort(combined_dist)

print("Query case:", query_id)
print("Semantic weight:", args.semantic_weight)
print("\nMost similar cases:")

shown = 0
for idx in ranking:
    if common_ids[idx] == query_id:
        continue

    print(
        common_ids[idx],
        "combined:", round(float(combined_dist[idx]), 4),
        "semantic:", round(float(semantic_dist[idx]), 4),
        "visual:", round(float(visual_dist[idx]), 4),
    )

    shown += 1
    if shown >= args.top_k:
        break