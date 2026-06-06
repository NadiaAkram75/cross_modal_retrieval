import nibabel as nib
import numpy as np

input_path = r"data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii"

volume = nib.load(input_path).get_fdata()

volume = (volume - volume.mean()) / volume.std()

np.save(
    "data/processed/BraTS20_Training_001.npy",
    volume.astype(np.float32)
)

print("Saved")