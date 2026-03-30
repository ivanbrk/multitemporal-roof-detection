import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from tqdm import tqdm


def _discover_tiffs(root_dir):
    root = Path(root_dir)
    tif_paths = list(root.rglob("*.tif")) + list(root.rglob("*.tiff"))
    return sorted({path.resolve() for path in tif_paths})


def _normalize_mask(mask):
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask[..., 0]
    mask = (mask > 0).astype(np.uint8)
    return mask


def _compute_mask_statistics(mask, pixel_size_m):
    mask = _normalize_mask(mask)
    fg_pixels = int(mask.sum())
    bg_pixels = int(mask.size - fg_pixels)
    fg_bg_ratio = float(fg_pixels) / float(max(bg_pixels, 1))
    labeled_mask, n_objects = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    del labeled_mask
    fg_area = float(fg_pixels) * float(pixel_size_m) * float(pixel_size_m)
    return fg_bg_ratio, int(n_objects), fg_area


def _collect_pairs(image_dir, mask_dir):
    image_paths = _discover_tiffs(image_dir)
    mask_paths = _discover_tiffs(mask_dir)

    image_index = {}
    for image_path in image_paths:
        stem = Path(image_path).stem
        image_index[stem] = str(image_path)

    mask_index = {}
    for mask_path in mask_paths:
        mask_stem = Path(mask_path).stem
        if not mask_stem.endswith("_mask"):
            continue
        tile_id = mask_stem[:-5]
        mask_index[tile_id] = str(mask_path)

    matched_tile_ids = sorted(set(image_index.keys()) & set(mask_index.keys()))
    masks_without_images = sorted(set(mask_index.keys()) - set(image_index.keys()))
    images_without_masks = sorted(set(image_index.keys()) - set(mask_index.keys()))

    pairs = []
    for tile_id in matched_tile_ids:
        pairs.append(
            {
                "tile_id": tile_id,
                "image_path": image_index[tile_id],
                "mask_path": mask_index[tile_id],
            }
        )

    if masks_without_images:
        print(
            "Warning: Skipping %d masks without a matching image. Examples: %s"
            % (
                len(masks_without_images),
                ", ".join("%s_mask" % tile_id for tile_id in masks_without_images[:5]),
            )
        )
    if images_without_masks:
        print(
            "Warning: Skipping %d images without a matching mask. Examples: %s"
            % (len(images_without_masks), ", ".join(images_without_masks[:5]))
        )

    if not pairs:
        raise RuntimeError("No valid image/mask pairs were found.")

    return sorted(pairs, key=lambda item: item["tile_id"])


def _assign_split(records, test_size, seed):
    if not 0.0 <= test_size <= 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    rng = np.random.RandomState(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)

    n_test = int(round(len(records) * test_size))
    if test_size > 0.0 and len(records) > 1:
        n_test = max(1, n_test)
    if test_size < 1.0 and len(records) > 1:
        n_test = min(len(records) - 1, n_test)

    test_indices = set(indices[:n_test].tolist())
    for idx, record in enumerate(records):
        record["train_test"] = "test" if idx in test_indices else "train"
    return records


def build_train_test_dataset(image_path, mask_path, output_xlsx, test_size, seed=42, pixel_size_m=0.5):
    records = _collect_pairs(image_path, mask_path)

    for record in tqdm(records, desc="Computing mask statistics", leave=True):
        mask = tifffile.imread(record["mask_path"])
        fg_bg_ratio, n_objects, fg_area = _compute_mask_statistics(mask, pixel_size_m)
        record["fg_bg_ratio"] = fg_bg_ratio
        record["n_objects"] = n_objects
        record["fg_area"] = fg_area

    records = _assign_split(records, test_size=test_size, seed=seed)
    dataframe = pd.DataFrame(records)
    dataframe = dataframe[
        [
            "tile_id",
            "image_path",
            "mask_path",
            "train_test",
            "fg_bg_ratio",
            "n_objects",
            "fg_area",
        ]
    ]

    output_dir = os.path.dirname(output_xlsx)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    dataframe.to_excel(output_xlsx, index=False)
    return dataframe


def parse_args():
    parser = argparse.ArgumentParser(description="Create the roofs_2026 XLSX train/test dataset file.")
    parser.add_argument("--image-path", required=True, help="Directory containing input images.")
    parser.add_argument("--mask-path", required=True, help="Directory containing input masks.")
    parser.add_argument("--output-xlsx", required=True, help="Output path for the XLSX train/test dataset file.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio between 0 and 1.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for the split.")
    parser.add_argument(
        "--pixel-size-m",
        type=float,
        default=0.5,
        help="Pixel size in meters. Foreground area is computed as pixel_size_m^2.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_train_test_dataset(
        image_path=args.image_path,
        mask_path=args.mask_path,
        output_xlsx=args.output_xlsx,
        test_size=args.test_size,
        seed=args.seed,
        pixel_size_m=args.pixel_size_m,
    )


if __name__ == "__main__":
    main()
