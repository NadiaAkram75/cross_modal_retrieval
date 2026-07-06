import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# -------------------
# LOAD EMBEDDINGS
# -------------------
embeddings = np.load("embeddings.npy")
case_ids = np.load("case_ids.npy", allow_pickle=True)

case_ids = [c[0] for c in case_ids]


# -------------------
# DIMENSION REDUCTION
# -------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca"
)

emb_2d = tsne.fit_transform(embeddings)


# -------------------
# PLOT
# -------------------
plt.figure(figsize=(10, 8))
plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=10)

for i in range(len(case_ids)):
    if i % 20 == 0:  # reduce clutter
        plt.text(emb_2d[i, 0], emb_2d[i, 1], case_ids[i], fontsize=6)

plt.title("t-SNE of MRI Embeddings")
plt.show()