import numpy as np

case_ids = np.load("semantic_case_ids.npy", allow_pickle=True)
features = np.load("semantic_features.npy")

wanted = [
    "BraTS20_Training_001",
    "BraTS20_Training_026",
    "BraTS20_Training_037",
    "BraTS20_Training_237",
    "BraTS20_Training_166",
    "BraTS20_Training_152",
]

print("case_id,total_tumor,necrotic,edema,enhancing,center_x,center_y,center_z,extent_mean")

for case_id in wanted:
    idx = list(case_ids).index(case_id)
    f = features[idx]
    print(
        case_id,
        round(f[0], 5),
        round(f[1], 5),
        round(f[2], 5),
        round(f[3], 5),
        round(f[7], 3),
        round(f[8], 3),
        round(f[9], 3),
        round(f[10], 3),
        sep=","
    )