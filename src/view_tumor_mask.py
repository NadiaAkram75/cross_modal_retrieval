import nibabel as nib
import matplotlib.pyplot as plt

base = r"data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001"

mri = nib.load(f"{base}/BraTS20_Training_001_t1ce.nii").get_fdata()
mask = nib.load(f"{base}/BraTS20_Training_001_seg.nii").get_fdata()

slice_idx = 77

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(mri[:, :, slice_idx], cmap="gray")
plt.title("MRI")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask[:, :, slice_idx])
plt.title("Tumor Mask")
plt.axis("off")

plt.show()