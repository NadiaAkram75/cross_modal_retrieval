import argparse
import csv
import os
import numpy as np


def clean_ids(ids):
    cleaned = []
    for x in ids:
        if isinstance(x, (list, np.ndarray)):
            cleaned.append(str(x[0]))
        else:
            cleaned.append(str(x))
    return cleaned


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


def evaluate_method(method_name, method_dist, semantic_dist, top_k, relevance_pool):
    mean_semantic_distances = []
    precision_scores = []
    recall_scores = []
    ndcg_scores = []

    n = method_dist.shape[0]

    for i in range(n):
        retrieved = ranking_without_self(method_dist, i)[:top_k]
        ideal = ranking_without_self(semantic_dist, i)

        relevant = set(ideal[:relevance_pool])
        retrieved_set = set(retrieved)

        overlap = len(retrieved_set.intersection(relevant))

        precision_scores.append(overlap / top_k)
        recall_scores.append(overlap / relevance_pool)

        semantic_distances = semantic_dist[i, retrieved]
        mean_semantic_distances.append(float(np.mean(semantic_distances)))

        gains = 1.0 / (1.0 + semantic_distances)
        ideal_gains = 1.0 / (1.0 + semantic_dist[i, ideal[:top_k]])

        ndcg = dcg(gains) / (dcg(ideal_gains) + 1e-8)
        ndcg_scores.append(ndcg)

    return {
        "method": method_name,
        "mean_semantic_distance_at_k": np.array(mean_semantic_distances),
        "precision_at_k": np.array(precision_scores),
        "recall_at_k": np.array(recall_scores),
        "ndcg_at_k": np.array(ndcg_scores),
    }


def bootstrap_ci(values, num_bootstrap=1000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)

    means = []
    for _ in range(num_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())

    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--relevance-pool", type=int, default=10)
    parser.add_argument("--weights", type=str, default="0,0.25,0.5,0.75,1.0")

    parser.add_argument(
        "--embedding-path",
        type=str,
        default="data/embeddings/embeddings.npy",
        help="Path to visual embeddings. Use contrastive embeddings to evaluate the contrastive encoder.",
    )
    parser.add_argument(
        "--case-id-path",
        type=str,
        default="data/embeddings/case_ids.npy",
        help="Path to case IDs corresponding to the visual embeddings.",
    )
    parser.add_argument(
        "--semantic-features-path",
        type=str,
        default="data/embeddings/semantic_features.npy",
        help="Path to semantic tumor-mask features.",
    )
    parser.add_argument(
        "--semantic-case-id-path",
        type=str,
        default="data/embeddings/semantic_case_ids.npy",
        help="Path to case IDs corresponding to semantic features.",
    )
    parser.add_argument(
        "--test-case-id-path",
        type=str,
        default=None,
        help="Optional path to test case IDs. If provided, evaluation is restricted to test cases.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/tables/retrieval_evaluation.csv",
        help="Output CSV path.",
    )

    args = parser.parse_args()

    embeddings = np.load(args.embedding_path)
    case_ids = clean_ids(np.load(args.case_id_path, allow_pickle=True))

    semantic_features = np.load(args.semantic_features_path)
    semantic_case_ids = clean_ids(
        np.load(args.semantic_case_id_path, allow_pickle=True)
    )

    visual_map = {case_id: embeddings[i] for i, case_id in enumerate(case_ids)}
    semantic_map = {
        case_id: semantic_features[i]
        for i, case_id in enumerate(semantic_case_ids)
    }

    common_ids = [case_id for case_id in case_ids if case_id in semantic_map]

    if args.test_case_id_path is not None:
        test_ids = set(clean_ids(np.load(args.test_case_id_path, allow_pickle=True)))
        common_ids = [case_id for case_id in common_ids if case_id in test_ids]

    if len(common_ids) == 0:
        raise ValueError(
            "No common case IDs found between visual embeddings, semantic features, and optional test split."
        )

    visual = np.array([visual_map[case_id] for case_id in common_ids])
    semantic = np.array([semantic_map[case_id] for case_id in common_ids])

    visual_dist = cosine_distance_matrix(visual)
    semantic_dist = euclidean_distance_matrix(zscore(semantic))

    visual_scaled = row_minmax(visual_dist)
    semantic_scaled = row_minmax(semantic_dist)

    weights = [float(w) for w in args.weights.split(",")]

    methods = {
        "visual_only": visual_scaled,
        "semantic_only_oracle": semantic_scaled,
    }

    for w in weights:
        methods[f"hybrid_semantic_weight_{w}"] = (
            w * semantic_scaled + (1.0 - w) * visual_scaled
        )

    results = []

    for method_name, method_dist in methods.items():
        scores = evaluate_method(
            method_name=method_name,
            method_dist=method_dist,
            semantic_dist=semantic_dist,
            top_k=args.top_k,
            relevance_pool=args.relevance_pool,
        )

        row = {
            "method": method_name,
            "embedding_path": args.embedding_path,
            "case_id_path": args.case_id_path,
            "semantic_features_path": args.semantic_features_path,
            "semantic_case_id_path": args.semantic_case_id_path,
            "test_case_id_path": args.test_case_id_path,
            "num_cases": len(common_ids),
            "top_k": args.top_k,
            "relevance_pool": args.relevance_pool,
        }

        for metric_name in [
            "mean_semantic_distance_at_k",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k",
        ]:
            mean, low, high = bootstrap_ci(scores[metric_name])
            row[f"{metric_name}_mean"] = mean
            row[f"{metric_name}_ci_low"] = low
            row[f"{metric_name}_ci_high"] = high

        results.append(row)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluated {len(common_ids)} common cases")
    print(f"Visual embeddings: {args.embedding_path}")
    print(f"Test split: {args.test_case_id_path}")
    print(f"Saved: {args.out}")
    print()

    for row in results:
        print(row["method"])
        print(
            "  mean semantic distance@k:",
            round(row["mean_semantic_distance_at_k_mean"], 4),
            "[",
            round(row["mean_semantic_distance_at_k_ci_low"], 4),
            ",",
            round(row["mean_semantic_distance_at_k_ci_high"], 4),
            "]",
        )
        print("  precision@k:", round(row["precision_at_k_mean"], 4))
        print("  recall@k:", round(row["recall_at_k_mean"], 4))
        print("  ndcg@k:", round(row["ndcg_at_k_mean"], 4))
        print()


if __name__ == "__main__":
    main()