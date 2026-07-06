import numpy as np

def clean_ids(ids):
    return [str(x[0]) if isinstance(x, (list, np.ndarray)) else str(x) for x in ids]

def normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

def cosine_distances(x):
    x = normalize(x)
    return 1 - np.dot(x, x.T)

TOP_K = 5

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

visual_dist = cosine_distances(visual)
semantic_dist = cosine_distances(semantic)

methods = {
    "visual_only": visual_dist,
    "semantic_only": semantic_dist,
    "hybrid_75_semantic": 0.75 * semantic_dist + 0.25 * visual_dist,
}

print("Method,Mean semantic distance of top-5,Median semantic distance of top-5")

for name, dist in methods.items():
    scores = []

    for i in range(len(common_ids)):
        ranking = np.argsort(dist[i])
        ranking = [j for j in ranking if j != i]
        top = ranking[:TOP_K]
        scores.append(semantic_dist[i, top].mean())

    print(name, round(float(np.mean(scores)), 4), round(float(np.median(scores)), 4), sep=",")

print("\nLower is better.")