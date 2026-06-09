import torch
import torch.nn as nn

from model import AutoEncoder
from dataset import BraTSDataset


from torch.utils.data import DataLoader

dataset = BraTSDataset("data/processed/t1ce")

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

model = AutoEncoder()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

#volume, case_id = dataset[0]

#x = torch.tensor(volume)

#x = x.unsqueeze(0)
#x = x.unsqueeze(0)

x, case_id = next(iter(dataloader))
x = x.unsqueeze(1)

for epoch in range(10):

    reconstruction, z = model(x)

    loss = criterion(reconstruction, x)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}: {loss.item():.4f}")