import numpy as np
import matplotlib.pyplot as plt

from dataset import BraTSDataset
from torch.utils.data import DataLoader


# -------------------
# LOAD DATA
# -------------------
dataset = BraTSDataset("data/processed/t1ce")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

cases = []
volumes = []

for x, case_id in loader:
    volumes.append(x.squeeze(0).numpy())
    cases.append(case_id[0])


# -------------------
# LOAD EMBEDDINGS
# -------------------
embeddings = np.load("embeddings.npy")
case_ids = np.load("case_ids.npy", allow_pickle=True)


# -------------------
# SIMPLE INDEX MAP
# -------------------
case_to_idx = {c: i for i, c in enumerate(cases)}


# -------------------
# QUERY CASE
# -------------------
query_case = case_ids[0][0]
query_idx = case_to_idx[query_case]


# cosine retrieval logic (same as before)
embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
query = embeddings_norm[query_idx]

scores = embeddings_norm @ query

topk = np.argsort(-scores)[:5]


# -------------------
# VISUALIZATION
# -------------------
def show_slice(volume, title):
    z = volume.shape[0] // 2
    plt.imshow(volume[z], cmap="gray")
    plt.title(title)
    plt.axis("off")


plt.figure(figsize=(10, 5))

for i, idx in enumerate(topk):
    plt.subplot(1, 5, i + 1)
    show_slice(volumes[idx], str(case_ids[idx][0]))

plt.tight_layout()
plt.show()