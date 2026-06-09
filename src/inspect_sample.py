from dataset import BraTSDataset

dataset = BraTSDataset("data/processed/t1ce")

volume, case_id = dataset[0]

print(volume.min())
print(volume.max())
print(volume.mean())
print(volume.std())
print(volume.nbytes / (1024 * 1024))