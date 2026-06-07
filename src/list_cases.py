from pathlib import Path

root = Path(
    "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)

cases = [p for p in root.iterdir() if p.is_dir()]

print("Number of cases:", len(cases))

print("\nFirst 5 cases:")
for case in cases[:5]:
    print(case.name)