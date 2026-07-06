import torch
import numpy as np

from model import AutoEncoder
from dataset import BraTSDataset
from torch.utils.data import DataLoader


# -------------------
# LOAD DATA
# -------------------
dataset = BraTSDataset("data/processed/t1ce")

loader = DataLoader(dataset, batch_size=1, shuffle=False)


# -------------------
# LOAD MODEL
# -------------------
model = AutoEncoder()
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()


# -------------------
# EXTRACT EMBEDDINGS
# -------------------
embeddings = []
case_ids = []

with torch.no_grad():
    for x, case_id in loader:

        x = x.unsqueeze(1)

        reconstruction, z = model(x)

        embeddings.append(z.squeeze().numpy())
        case_ids.append(case_id)


# -------------------
# SAVE
# -------------------
embeddings = np.array(embeddings)

np.save("embeddings.npy", embeddings)
np.save("case_ids.npy", np.array(case_ids))

print("Embeddings saved:", embeddings.shape)