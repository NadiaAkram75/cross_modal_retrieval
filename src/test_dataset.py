from dataset import BraTSDataset

dataset = BraTSDataset("data/processed/t1ce")

print(len(dataset))
volume, case_id = dataset[0]

print(case_id)
print(volume.shape)