from dataset import BraTSDataset
from model import SimpleEncoder
import torch
from model import SimpleDecoder
from model import AutoEncoder

dataset = BraTSDataset("data/processed/t1ce")

volume, case_id = dataset[0]

x = torch.tensor(volume)

x = x.unsqueeze(0)
x = x.unsqueeze(0)

model = SimpleEncoder()

y = model(x)


print(x.shape)
print(y.shape)
print(y[0])
print(y)




print("---------------------------------------------------------decoder---------------------------------------------------------")
decoder = SimpleDecoder()

z = torch.randn(1, 32)

output = decoder(z)

print(output.shape)



print("---------------------------------------------------------AutoEncoder---------------------------------------------------------")


model = AutoEncoder()

reconstruction, z = model(x)

print(reconstruction.shape)
print(z.shape)