import nibabel as nib
import numpy as np

path = r"data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii"

volume = nib.load(path).get_fdata()

volume = (volume - volume.mean()) / volume.std()

print("Mean:", np.mean(volume))
print("Std:", np.std(volume))