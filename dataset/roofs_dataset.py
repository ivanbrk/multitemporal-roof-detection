import pandas as pd
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset


def _channels_last(image):
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
        return image

    if image.ndim != 3:
        raise ValueError("Expected a 2D or 3D TIFF image, got %s." % (image.shape,))

    if image.shape[-1] in (1, 3, 4):
        return image
    if image.shape[0] in (1, 3, 4):
        return np.transpose(image, (1, 2, 0))
    raise ValueError("Unable to infer channel order for shape %s." % (image.shape,))


def load_tif_image(path):
    image = tifffile.imread(path)
    image = _channels_last(image)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]

    original_dtype = image.dtype
    image = image.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        dtype_info = np.iinfo(original_dtype)
        if dtype_info.max > 0:
            image /= float(dtype_info.max)
    else:
        image = np.clip(image, 0.0, None)
        max_value = float(image.max()) if image.size else 1.0
        if max_value > 1.0:
            image /= max_value

    return np.clip(image, 0.0, 1.0)


def load_tif_mask(path):
    mask = tifffile.imread(path)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask[..., 0]
    mask = (mask > 0).astype(np.float32)
    return mask


class RoofDataset(Dataset):
    def __init__(
        self,
        train_test_dataset_path,
        split="train",
        transforms=None,
        inference=False,
        selected_tiles=None,
        limit=None,
    ):
        dataframe = pd.read_excel(train_test_dataset_path)
        if split in ("train", "test"):
            dataframe = dataframe[dataframe["train_test"] == split]
        elif split not in ("all", None):
            raise ValueError("Unknown split: %s" % split)

        if selected_tiles is not None:
            dataframe = dataframe[dataframe["tile_id"].isin(selected_tiles)]

        dataframe = dataframe.sort_values("tile_id").reset_index(drop=True)
        if limit is not None:
            dataframe = dataframe.head(int(limit)).reset_index(drop=True)

        self.dataframe = dataframe
        self.records = dataframe.to_dict("records")
        self.transforms = transforms
        self.inference = inference

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = load_tif_image(record["image_path"])

        if self.inference:
            if self.transforms is not None:
                transformed = self.transforms(image=image)
                image = transformed["image"]
            image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
            return {
                "image": image_tensor,
                "tile_id": record["tile_id"],
                "image_path": record["image_path"],
            }

        mask = load_tif_mask(record["mask_path"])
        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_tensor = torch.from_numpy(mask[None, ...]).float()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "tile_id": record["tile_id"],
            "image_path": record["image_path"],
            "mask_path": record["mask_path"],
        }
