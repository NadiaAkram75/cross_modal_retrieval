import nibabel as nib
import matplotlib.pyplot as plt

path = r"data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_t1ce.nii"

img = nib.load(path)
volume = img.get_fdata()

middle_slice = volume[:, :, 77]

plt.imshow(middle_slice, cmap="gray")
plt.axis("off")
plt.show()