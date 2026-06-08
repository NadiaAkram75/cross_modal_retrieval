from dataset import BraTSDataset

dataset = BraTSDataset("data/processed/t1ce")

print(len(dataset))
print(dataset[0].shape)