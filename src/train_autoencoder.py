import torch
import torch.nn as nn
import torch.nn.functional as F

from model import AutoEncoder
from dataset import BraTSDataset

from torch.utils.data import DataLoader, random_split


# -------------------
# DATASET
# -------------------
dataset = BraTSDataset("data/processed/t1ce")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


# -------------------
# MODEL
# -------------------
model = AutoEncoder()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

best_loss = float("inf")


# -------------------
# TRAINING
# -------------------
for epoch in range(10):

    model.train()
    train_loss = 0

    for x, case_id in train_loader:

        x = x.unsqueeze(1)

        # -------------------
        # two augmented views
        # -------------------
        x1 = x + 0.01 * torch.randn_like(x)
        x2 = x + 0.01 * torch.randn_like(x)

        _, z1 = model(x1)
        _, z2 = model(x2)

        # -------------------
        # normalize embeddings
        # -------------------
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # -------------------
        # contrastive objective (positive pair)
        # -------------------
        similarity = (z1 * z2).sum(dim=1)
        loss = 1 - similarity.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train contrastive loss {avg_train_loss:.4f}")


    # -------------------
    # VALIDATION (same objective)
    # -------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, case_id in val_loader:

            x = x.unsqueeze(1)

            x1 = x + 0.01 * torch.randn_like(x)
            x2 = x + 0.01 * torch.randn_like(x)

            _, z1 = model(x1)
            _, z2 = model(x2)

            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)

            similarity = (z1 * z2).sum(dim=1)
            loss = 1 - similarity.mean()

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Val contrastive loss {avg_val_loss:.4f}")


    # -------------------
    # SAVE BEST MODEL
    # -------------------
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved best model")