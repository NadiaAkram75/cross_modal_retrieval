import nibabel as nib
import numpy as np

path = r"data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_seg.nii"

mask = nib.load(path).get_fdata()

print(np.unique(mask))