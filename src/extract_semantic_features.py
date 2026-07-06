import os
import numpy as np
import nibabel as nib

DATA_ROOT = r"data\raw\brats\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

case_ids = np.load("case_ids.npy", allow_pickle=True)

features = []
valid_case_ids = []

for item in case_ids:
    case_id = str(item[0]) if isinstance(item, (list, np.ndarray)) else str(item)

    seg_path = os.path.join(DATA_ROOT, case_id, f"{case_id}_seg.nii")

    if not os.path.exists(seg_path):
        print("Missing mask:", case_id)
        continue

    seg = nib.load(seg_path).get_fdata()

    tumor = seg > 0
    label_1 = seg == 1   # necrotic / non-enhancing tumor
    label_2 = seg == 2   # edema
    label_4 = seg == 4   # enhancing tumor

    total = tumor.sum()

    if total == 0:
        feat = np.zeros(12)
    else:
        coords = np.argwhere(tumor)

        centroid = coords.mean(axis=0) / np.array(seg.shape)
        bbox_min = coords.min(axis=0)
        bbox_max = coords.max(axis=0)
        extent = (bbox_max - bbox_min + 1) / np.array(seg.shape)

        feat = np.array([
            total / seg.size,
            label_1.sum() / seg.size,
            label_2.sum() / seg.size,
            label_4.sum() / seg.size,
            label_1.sum() / total,
            label_2.sum() / total,
            label_4.sum() / total,
            centroid[0],
            centroid[1],
            centroid[2],
            extent.mean(),
            extent.max()
        ])

    features.append(feat)
    valid_case_ids.append(case_id)

features = np.array(features)
valid_case_ids = np.array(valid_case_ids)

np.save("semantic_features.npy", features)
np.save("semantic_case_ids.npy", valid_case_ids)

print("Saved semantic features:", features.shape)
print("Saved semantic case ids:", valid_case_ids.shape)