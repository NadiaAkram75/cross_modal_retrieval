import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

DATA_ROOT = r"data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
CASE_IDS = [
    "BraTS20_Training_001",
    "BraTS20_Training_026",
    "BraTS20_Training_037",
    "BraTS20_Training_237",
    "BraTS20_Training_166",
    "BraTS20_Training_152",
]

plt.figure(figsize=(15, 3))

for i, case_id in enumerate(CASE_IDS):
    flair_path = os.path.join(DATA_ROOT, case_id, f"{case_id}_flair.nii")
    seg_path = os.path.join(DATA_ROOT, case_id, f"{case_id}_seg.nii")

    flair = nib.load(flair_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()

    z = flair.shape[2] // 2

    plt.subplot(1, len(CASE_IDS), i + 1)
    plt.imshow(flair[:, :, z], cmap="gray")
    plt.imshow(seg[:, :, z] > 0, alpha=0.35)
    plt.title(("Query\n" if i == 0 else "Retrieved\n") + case_id, fontsize=8)
    plt.axis("off")

plt.tight_layout()
plt.savefig("semantic_mask_overlay_result.png", dpi=200)
print("Saved: semantic_mask_overlay_result.png")