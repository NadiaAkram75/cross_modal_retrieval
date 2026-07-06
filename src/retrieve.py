import numpy as np
from sklearn.neighbors import NearestNeighbors


# -------------------
# LOAD EMBEDDINGS
# -------------------
embeddings = np.load("embeddings.npy")
case_ids = np.load("case_ids.npy", allow_pickle=True)


# -------------------
# NORMALIZE (key change)
# -------------------
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


# -------------------
# COSINE NEIGHBORS
# -------------------
nn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
nn_model.fit(embeddings)


# -------------------
# QUERY
# -------------------
query = embeddings[0].reshape(1, -1)

distances, indices = nn_model.kneighbors(query)


print("Query case:", case_ids[0])
print("\nMost similar cases (cosine):")

for i, idx in enumerate(indices[0]):
    print(case_ids[idx], "distance:", distances[0][i])