import argparse
import csv
import os
import numpy as np


FEATURE_NAMES = [
    "tumor_volume_fraction_brain",
    "necrotic_fraction_brain",
    "edema_fraction_brain",
    "enhancing_fraction_brain",
    "necrotic_fraction_tumor",
    "edema_fraction_tumor",
    "enhancing_fraction_tumor",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "extent_mean",
    "extent_max",
]


DISPLAY_FEATURES = [
    "tumor_volume_fraction_brain",
    "necrotic_fraction_tumor",
    "edema_fraction_tumor",
    "enhancing_fraction_tumor",
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "extent_mean",
    "extent_max",
]


def clean_ids(ids):
    return [str(x[0]) if isinstance(x, (list, np.ndarray)) else str(x) for x in ids]


def l2_normalize(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def cosine_distance_matrix(x):
    x = l2_normalize(x)
    return 1.0 - np.dot(x, x.T)


def euclidean_distance_matrix(x):
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def row_minmax(d):
    scaled = np.zeros_like(d)
    for i in range(d.shape[0]):
        row = d[i].copy()
        mask = np.ones(len(row), dtype=bool)
        mask[i] = False
        mn = row[mask].min()
        mx = row[mask].max()
        scaled[i] = (row - mn) / (mx - mn + 1e-8)
        scaled[i, i] = 0.0
    return scaled


def ranking_without_self(dist_matrix, i):
    ranking = np.argsort(dist_matrix[i])
    return [j for j in ranking if j != i]


def summarize_similarity(query_f, retrieved_f):
    diffs = np.abs(query_f - retrieved_f)
    selected = []

    for name in DISPLAY_FEATURES:
        idx = FEATURE_NAMES.index(name)
        selected.append((name, diffs[idx]))

    selected = sorted(selected, key=lambda x: x[1])
    return "; ".join([f"{name} close" for name, _ in selected[:3]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-case", type=str, default=None)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-weight", type=float, default=0.75)
    parser.add_argument("--out", type=str, default="results/retrieval_explanation.csv")
    args = parser.parse_args()

    embeddings = np.load("embeddings.npy")
    case_ids = clean_ids(np.load("case_ids.npy", allow_pickle=True))

    semantic_features = np.load("semantic_features.npy")
    semantic_case_ids = clean_ids(np.load("semantic_case_ids.npy", allow_pickle=True))

    visual_map = {case_id: embeddings[i] for i, case_id in enumerate(case_ids)}
    semantic_map = {
        case_id: semantic_features[i]
        for i, case_id in enumerate(semantic_case_ids)
    }

    common_ids = [case_id for case_id in case_ids if case_id in semantic_map]

    visual = np.array([visual_map[case_id] for case_id in common_ids])
    semantic = np.array([semantic_map[case_id] for case_id in common_ids])

    visual_dist = row_minmax(cosine_distance_matrix(visual))
    semantic_dist = row_minmax(euclidean_distance_matrix(zscore(semantic)))

    hybrid_dist = (
        args.semantic_weight * semantic_dist
        + (1.0 - args.semantic_weight) * visual_dist
    )

    if args.query_case is not None:
        query_index = common_ids.index(args.query_case)
    else:
        query_index = args.query_index

    query_id = common_ids[query_index]
    query_f = semantic[query_index]

    top_indices = ranking_without_self(hybrid_dist, query_index)[:args.top_k]

    rows = []

    for rank, idx in enumerate(top_indices, start=1):
        retrieved_id = common_ids[idx]
        retrieved_f = semantic[idx]

        row = {
            "query_case": query_id,
            "rank": rank,
            "retrieved_case": retrieved_id,
            "hybrid_distance": float(hybrid_dist[query_index, idx]),
            "semantic_distance": float(semantic_dist[query_index, idx]),
            "visual_distance": float(visual_dist[query_index, idx]),
            "main_similarity_explanation": summarize_similarity(query_f, retrieved_f),
        }

        for feature_name in DISPLAY_FEATURES:
            feature_idx = FEATURE_NAMES.index(feature_name)
            row[f"query_{feature_name}"] = float(query_f[feature_idx])
            row[f"retrieved_{feature_name}"] = float(retrieved_f[feature_idx])
            row[f"absolute_difference_{feature_name}"] = float(
                abs(query_f[feature_idx] - retrieved_f[feature_idx])
            )

        rows.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Query case: {query_id}")
    print(f"Saved explanation table: {args.out}")
    print()

    for row in rows:
        print(
            f"Rank {row['rank']}: {row['retrieved_case']} | "
            f"semantic distance={row['semantic_distance']:.3f} | "
            f"{row['main_similarity_explanation']}"
        )


if __name__ == "__main__":
    main()