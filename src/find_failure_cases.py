import argparse
import csv
import os
import numpy as np


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


def dcg(gains):
    gains = np.asarray(gains)
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains * discounts))


def ndcg_for_query(query_index, retrieved_indices, semantic_dist):
    gains = 1.0 / (1.0 + semantic_dist[query_index, retrieved_indices])

    ideal = ranking_without_self(semantic_dist, query_index)[: len(retrieved_indices)]
    ideal_gains = 1.0 / (1.0 + semantic_dist[query_index, ideal])

    return dcg(gains) / (dcg(ideal_gains) + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--relevance-pool", type=int, default=10)
    parser.add_argument("--semantic-weight", type=float, default=0.75)
    parser.add_argument("--num-failures", type=int, default=20)
    parser.add_argument("--out", type=str, default="results/failure_cases.csv")
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

    rows = []

    for i, query_case in enumerate(common_ids):
        hybrid_top = ranking_without_self(hybrid_dist, i)[: args.top_k]
        semantic_oracle_top = ranking_without_self(semantic_dist, i)[: args.top_k]
        relevant_pool = set(
            ranking_without_self(semantic_dist, i)[: args.relevance_pool]
        )

        overlap = len(set(hybrid_top).intersection(relevant_pool))

        hybrid_mean_semantic_distance = float(np.mean(semantic_dist[i, hybrid_top]))
        oracle_mean_semantic_distance = float(
            np.mean(semantic_dist[i, semantic_oracle_top])
        )

        failure_gap = hybrid_mean_semantic_distance - oracle_mean_semantic_distance

        row = {
            "query_case": query_case,
            "hybrid_mean_semantic_distance_at_k": hybrid_mean_semantic_distance,
            "oracle_mean_semantic_distance_at_k": oracle_mean_semantic_distance,
            "failure_gap_vs_oracle": failure_gap,
            "precision_at_k": overlap / args.top_k,
            "recall_at_k": overlap / args.relevance_pool,
            "ndcg_at_k": ndcg_for_query(i, hybrid_top, semantic_dist),
            "hybrid_retrieved_cases": "; ".join([common_ids[j] for j in hybrid_top]),
            "semantic_oracle_cases": "; ".join(
                [common_ids[j] for j in semantic_oracle_top]
            ),
            "note": (
                "Semantic distance is row-wise min-max normalized. "
                "A value of 0.000 means closest normalized match, not identical anatomy."
            ),
        }

        rows.append(row)

    rows = sorted(
        rows,
        key=lambda x: (
            x["failure_gap_vs_oracle"],
            x["hybrid_mean_semantic_distance_at_k"],
        ),
        reverse=True,
    )

    rows = rows[: args.num_failures]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved worst {len(rows)} failure cases to: {args.out}")
    print()

    for row in rows[:10]:
        print(
            f"{row['query_case']} | "
            f"gap={row['failure_gap_vs_oracle']:.3f} | "
            f"hybrid sem dist@k={row['hybrid_mean_semantic_distance_at_k']:.3f} | "
            f"precision@k={row['precision_at_k']:.3f} | "
            f"nDCG@k={row['ndcg_at_k']:.3f}"
        )


if __name__ == "__main__":
    main()