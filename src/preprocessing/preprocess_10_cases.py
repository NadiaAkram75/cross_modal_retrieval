from pathlib import Path
import nibabel as nib
import numpy as np

root = Path(
    "data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
)

output_dir = Path("data/processed/t1ce")
output_dir.mkdir(parents=True, exist_ok=True)

cases = sorted([p for p in root.iterdir() if p.is_dir()])[:10]

for case in cases:
    case_id = case.name

    t1ce_file = case / f"{case_id}_t1ce.nii"

    volume = nib.load(str(t1ce_file)).get_fdata()

    volume = (volume - volume.mean()) / volume.std()

    np.save(
        output_dir / f"{case_id}.npy",
        volume.astype(np.float32)
    )

    print(f"Processed {case_id}")

print("Done")