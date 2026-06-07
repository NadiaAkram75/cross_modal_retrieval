import numpy as np
from monai.transforms import Resize
import os


print(os.getcwd())

os.makedirs("data/processed/test_resize", exist_ok=True)

volume = np.load(
    "data/processed/BraTS20_Training_001.npy"
)

print(volume.shape)
resize = Resize(spatial_size=(128, 128, 128))

resized = resize(volume[np.newaxis, ...])

resized = resized.squeeze(0)

print(resized.shape)

np.save(
    "data/processed/test_resize/BraTS20_Training_001.npy",
    resized.numpy()
)