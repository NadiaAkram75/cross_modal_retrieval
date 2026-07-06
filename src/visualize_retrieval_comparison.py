import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

DATA_ROOT = r"data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"


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


def find_nifti(case_id, suffix):
    candidates = [
        os.path.join(DATA_ROOT, case_id, f"{case_id}_{suffix}.nii"),
        os.path.join(DATA_ROOT, case_id, f"{case_id}_{suffix}.nii.gz"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Missing {suffix} file for {case_id}")


def load_best_slice(case_id):
    flair_path = find_nifti(case_id, "flair")
    seg_path = find_nifti(case_id, "seg")

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    tumor_pixels_per_slice = (seg > 0).sum(axis=(0, 1))

    if tumor_pixels_per_slice.max() > 0:
        z = int(np.argmax(tumor_pixels_per_slice))
    else:
        z = flair.shape[2] // 2

    image = flair[:, :, z]
    mask = seg[:, :, z] > 0

    low, high = np.percentile(image, [1, 99])
    image = np.clip(image, low, high)
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return image, mask


def select_interesting_query(visual_dist, hybrid_dist, semantic_dist, common_ids, top_k):
    best_score = -np.inf
    best_index = 0

    for i, case_id in enumerate(common_ids):
        try:
            find_nifti(case_id, "flair")
            find_nifti(case_id, "seg")
        except FileNotFoundError:
            continue

        visual_top = ranking_without_self(visual_dist, i)[:top_k]
        hybrid_top = ranking_without_self(hybrid_dist, i)[:top_k]

        visual_semantic_error = semantic_dist[i, visual_top].mean()
        hybrid_semantic_error = semantic_dist[i, hybrid_top].mean()

        score = visual_semantic_error - hybrid_semantic_error

        if score > best_score:
            best_score = score
            best_index = i

    return best_index


def plot_case(ax, case_id, title):
    image, mask = load_best_slice(case_id)

    ax.imshow(image, cmap="gray")

    if mask.sum() > 0:
        ax.contour(mask, levels=[0.5], colors="red", linewidths=1)

    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-index", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-weight", type=float, default=0.75)
    parser.add_argument("--out", type=str, default="results/retrieval_comparison.png")
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

    if args.query_index is None:
        query_index = select_interesting_query(
            visual_dist=visual_dist,
            hybrid_dist=hybrid_dist,
            semantic_dist=semantic_dist,
            common_ids=common_ids,
            top_k=args.top_k,
        )
    else:
        query_index = args.query_index

    query_id = common_ids[query_index]

    visual_top = ranking_without_self(visual_dist, query_index)[:args.top_k]
    hybrid_top = ranking_without_self(hybrid_dist, query_index)[:args.top_k]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fig, axes = plt.subplots(
        2,
        args.top_k + 1,
        figsize=(3 * (args.top_k + 1), 6),
    )

    plot_case(
        axes[0, 0],
        query_id,
        f"Query\n{query_id}",
    )
    plot_case(
        axes[1, 0],
        query_id,
        f"Query\n{query_id}",
    )

    axes[0, 0].set_ylabel("Visual-only", fontsize=11)
    axes[1, 0].set_ylabel(
        f"Hybrid\nsemantic weight={args.semantic_weight}",
        fontsize=11,
    )

    for rank, idx in enumerate(visual_top, start=1):
        case_id = common_ids[idx]
        sem_d = semantic_dist[query_index, idx]
        plot_case(
            axes[0, rank],
            case_id,
            f"Rank {rank}\n{case_id}\nsem d={sem_d:.3f}",
        )

    for rank, idx in enumerate(hybrid_top, start=1):
        case_id = common_ids[idx]
        sem_d = semantic_dist[query_index, idx]
        plot_case(
            axes[1, rank],
            case_id,
            f"Rank {rank}\n{case_id}\nsem d={sem_d:.3f}",
        )

    plt.suptitle(
        "Retrieval comparison: visual-only vs hybrid semantic retrieval",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=250)
    plt.close()

    print(f"Query case: {query_id}")
    print(f"Saved figure: {args.out}")


if __name__ == "__main__":
    main()