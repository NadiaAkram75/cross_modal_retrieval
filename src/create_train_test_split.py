import argparse
import os
import numpy as np
import pandas as pd


def clean_ids(ids):
    cleaned = []

    for x in ids:
        if isinstance(x, (list, np.ndarray)):
            cleaned.append(str(x[0]))
        else:
            cleaned.append(str(x))

    return cleaned


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--semantic-case-id-path",
        type=str,
        default="data/embeddings/semantic_case_ids.npy",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="data/splits")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    case_ids = clean_ids(np.load(args.semantic_case_id_path, allow_pickle=True))
    case_ids = np.array(sorted(case_ids), dtype=object)

    rng = np.random.default_rng(args.seed)
    shuffled = case_ids.copy()
    rng.shuffle(shuffled)

    n_test = int(round(len(shuffled) * args.test_size))

    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]

    np.save(os.path.join(args.out_dir, "train_case_ids.npy"), train_ids)
    np.save(os.path.join(args.out_dir, "test_case_ids.npy"), test_ids)

    split_df = pd.DataFrame(
        {
            "case_id": list(train_ids) + list(test_ids),
            "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
        }
    )

    split_df.to_csv(os.path.join(args.out_dir, "case_split.csv"), index=False)

    print(f"Total cases: {len(case_ids)}")
    print(f"Train cases: {len(train_ids)}")
    print(f"Test cases: {len(test_ids)}")
    print(f"Saved split files to: {args.out_dir}")


if __name__ == "__main__":
    main()