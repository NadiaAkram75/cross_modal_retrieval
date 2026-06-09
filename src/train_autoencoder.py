import torch
import torch.nn as nn

from model import AutoEncoder
from dataset import BraTSDataset

dataset = BraTSDataset("data/processed/t1ce")

model = AutoEncoder()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

volume, case_id = dataset[0]

x = torch.tensor(volume)

x = x.unsqueeze(0)
x = x.unsqueeze(0)

# reconstruction, z = model(x)

# print("input:", x.shape)
# print("reconstruction:", reconstruction.shape)
# print("embedding:", z.shape)

# loss = criterion(reconstruction, x)

# print("loss:", loss.item())

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

# reconstruction, z = model(x)

# new_loss = criterion(reconstruction, x)

# print("new loss:", new_loss.item())

for epoch in range(10):

    reconstruction, z = model(x)

    loss = criterion(reconstruction, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: {loss.item():.4f}")