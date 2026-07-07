import argparse
import os
import numpy as np
import torch

from train_contrastive_encoder import (
    DEFAULT_DATA_ROOT,
    first_existing,
    SemanticPairDataset,
    ContrastiveEncoder,
    extract_embeddings,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/contrastive_encoder_train_only.pth",
    )
    parser.add_argument(
        "--split-case-id-path",
        type=str,
        default="data/splits/test_case_ids.npy",
    )
    parser.add_argument(
        "--embedding-out",
        type=str,
        default="data/embeddings/contrastive_test_embeddings.npy",
    )
    parser.add_argument(
        "--case-id-out",
        type=str,
        default="data/embeddings/contrastive_test_case_ids.npy",
    )

    args = parser.parse_args()

    semantic_features_path = first_existing(
        [
            "data/embeddings/semantic_features.npy",
            "semantic_features.npy",
        ]
    )

    semantic_case_ids_path = first_existing(
        [
            "data/embeddings/semantic_case_ids.npy",
            "semantic_case_ids.npy",
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device)

    embedding_dim = checkpoint.get("embedding_dim", 128)
    image_size = checkpoint.get("image_size", 128)

    dataset = SemanticPairDataset(
        data_root=args.data_root,
        semantic_features_path=semantic_features_path,
        semantic_case_ids_path=semantic_case_ids_path,
        image_size=image_size,
        positive_k=10,
        split_case_id_path=args.split_case_id_path,
    )

    model = ContrastiveEncoder(embedding_dim=embedding_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    embeddings, case_ids = extract_embeddings(model, dataset, device)

    os.makedirs(os.path.dirname(args.embedding_out), exist_ok=True)

    np.save(args.embedding_out, embeddings)
    np.save(args.case_id_out, case_ids)

    print(f"Saved embeddings: {args.embedding_out}")
    print(f"Saved case IDs: {args.case_id_out}")
    print(f"Number of cases: {len(case_ids)}")


if __name__ == "__main__":
    main()