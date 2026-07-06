import argparse
import os
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DEFAULT_DATA_ROOT = (
    "data/raw/brats/BraTS2020_TrainingData/"
    "MICCAI_BraTS2020_TrainingData"
)


def first_existing(paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these paths exist: {paths}")


def clean_ids(ids):
    cleaned = []
    for x in ids:
        if isinstance(x, (list, np.ndarray)):
            cleaned.append(str(x[0]))
        else:
            cleaned.append(str(x))
    return cleaned


def zscore(x):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-8)


def find_nifti(data_root, case_id, suffix):
    candidates = [
        os.path.join(data_root, case_id, f"{case_id}_{suffix}.nii"),
        os.path.join(data_root, case_id, f"{case_id}_{suffix}.nii.gz"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Missing {suffix} file for {case_id}")


def load_best_flair_slice(data_root, case_id, image_size):
    flair_path = find_nifti(data_root, case_id, "flair")
    seg_path = find_nifti(data_root, case_id, "seg")

    flair = nib.load(flair_path).get_fdata(dtype=np.float32)
    seg = nib.load(seg_path).get_fdata(dtype=np.float32)

    tumor_pixels_per_slice = (seg > 0).sum(axis=(0, 1))

    if tumor_pixels_per_slice.max() > 0:
        z = int(np.argmax(tumor_pixels_per_slice))
    else:
        z = flair.shape[2] // 2

    image = flair[:, :, z].astype(np.float32)

    low, high = np.percentile(image, [1, 99])
    image = np.clip(image, low, high)

    image_min = image.min()
    image_max = image.max()
    image = (image - image_min) / (image_max - image_min + 1e-8)
    image = image.astype(np.float32)

    image = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)

    image = F.interpolate(
        image,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    return image.squeeze(0).float()


class SemanticPairDataset(Dataset):
    def __init__(
        self,
        data_root,
        semantic_features_path,
        semantic_case_ids_path,
        image_size=128,
        positive_k=10,
    ):
        self.data_root = data_root
        self.image_size = image_size
        self.positive_k = positive_k

        semantic_features = np.load(semantic_features_path)
        semantic_case_ids = clean_ids(
            np.load(semantic_case_ids_path, allow_pickle=True)
        )

        valid_case_ids = []
        valid_features = []

        for case_id, feature in zip(semantic_case_ids, semantic_features):
            try:
                find_nifti(data_root, case_id, "flair")
                find_nifti(data_root, case_id, "seg")
                valid_case_ids.append(case_id)
                valid_features.append(feature)
            except FileNotFoundError:
                continue

        self.case_ids = valid_case_ids
        self.semantic_features = np.array(valid_features, dtype=np.float32)

        semantic_z = zscore(self.semantic_features)
        diff = semantic_z[:, None, :] - semantic_z[None, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=2))

        self.positive_indices = []

        for i in range(len(self.case_ids)):
            ranking = np.argsort(dist[i])
            ranking = [j for j in ranking if j != i]
            self.positive_indices.append(ranking[:positive_k])

        self.image_cache = {}

        print(f"Loaded {len(self.case_ids)} valid cases")

    def __len__(self):
        return len(self.case_ids)

    def get_image(self, index):
        if index not in self.image_cache:
            case_id = self.case_ids[index]
            self.image_cache[index] = load_best_flair_slice(
                self.data_root,
                case_id,
                self.image_size,
            )
        return self.image_cache[index].float()

    def __getitem__(self, index):
        positive_index = random.choice(self.positive_indices[index])

        anchor = self.get_image(index)
        positive = self.get_image(positive_index)

        return anchor, positive, index, positive_index


class ContrastiveEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        x = x.float()
        x = self.backbone(x)
        x = x.flatten(1)
        x = self.projector(x)
        x = F.normalize(x, dim=1)
        return x


def contrastive_loss(z1, z2, temperature=0.1):
    batch_size = z1.shape[0]

    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    self_mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(self_mask, -1e9)

    positive_indices = torch.arange(2 * batch_size, device=z.device)
    positive_indices = (positive_indices + batch_size) % (2 * batch_size)

    loss = -sim[torch.arange(2 * batch_size), positive_indices]
    loss += torch.logsumexp(sim, dim=1)

    return loss.mean()


@torch.no_grad()
def extract_embeddings(model, dataset, device):
    model.eval()

    embeddings = []
    case_ids = []

    for i in range(len(dataset)):
        image = dataset.get_image(i).unsqueeze(0).to(device).float()
        embedding = model(image).cpu().numpy()[0]

        embeddings.append(embedding)
        case_ids.append(dataset.case_ids[i])

    return np.array(embeddings, dtype=np.float32), np.array(case_ids, dtype=object)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--positive-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--model-out", type=str, default="models/contrastive_encoder.pth")
    parser.add_argument(
        "--embedding-out",
        type=str,
        default="data/embeddings/contrastive_embeddings.npy",
    )
    parser.add_argument(
        "--case-id-out",
        type=str,
        default="data/embeddings/contrastive_case_ids.npy",
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

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = SemanticPairDataset(
        data_root=args.data_root,
        semantic_features_path=semantic_features_path,
        semantic_case_ids_path=semantic_case_ids_path,
        image_size=args.image_size,
        positive_k=args.positive_k,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    model = ContrastiveEncoder(embedding_dim=args.embedding_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for anchor, positive, _, _ in loader:
            anchor = anchor.to(device).float()
            positive = positive.to(device).float()

            z1 = model(anchor)
            z2 = model(positive)

            loss = contrastive_loss(z1, z2, temperature=args.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch:03d} | loss={np.mean(losses):.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "embedding_dim": args.embedding_dim,
            "image_size": args.image_size,
        },
        args.model_out,
    )

    embeddings, case_ids = extract_embeddings(model, dataset, device)

    np.save(args.embedding_out, embeddings)
    np.save(args.case_id_out, case_ids)

    print(f"Saved model: {args.model_out}")
    print(f"Saved embeddings: {args.embedding_out}")
    print(f"Saved case ids: {args.case_id_out}")


if __name__ == "__main__":
    main()