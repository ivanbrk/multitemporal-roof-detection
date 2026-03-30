import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset.roofs_dataset import load_tif_image, load_tif_mask


YELLOW_LINE_WIDTH = 6


def _use_progress_bar():
    return sys.stderr.isatty()


def _to_uint8_rgb(image):
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def _mask_to_rgb(mask):
    mask = (mask > 0).astype(np.uint8) * 255
    return np.stack([mask, mask, mask], axis=-1)


def _save_image(array, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    Image.fromarray(array).save(path)


def _combine_side_by_side(left, right, separator_color=(255, 255, 0)):
    height = left.shape[0]
    separator = np.zeros((height, YELLOW_LINE_WIDTH, 3), dtype=np.uint8)
    separator[:, :] = np.array(separator_color, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def save_augmentation_examples(records, transform, output_dir, max_examples):
    os.makedirs(output_dir, exist_ok=True)
    selected_records = records[: max(0, int(max_examples))]
    if _use_progress_bar():
        iterable = tqdm(selected_records, desc="Saving augmentation visualizations", leave=True)
    else:
        iterable = selected_records
        print("Saving %d augmentation visualizations to %s." % (len(selected_records), output_dir), flush=True)

    for record in iterable:
        image = load_tif_image(record["image_path"])
        mask = load_tif_mask(record["mask_path"])
        transformed = transform(image=image, mask=mask)
        augmented_image = _to_uint8_rgb(transformed["image"])
        augmented_mask = _mask_to_rgb(transformed["mask"])
        canvas = _combine_side_by_side(augmented_image, augmented_mask, separator_color=(255, 255, 0))
        filename = "%s_aug.png" % record["tile_id"]
        _save_image(canvas, os.path.join(output_dir, filename))

    if not _use_progress_bar():
        print("Saved %d augmentation visualizations." % len(selected_records), flush=True)


def _build_confusion_overlay(image, pred_mask, gt_mask):
    image_rgb = _to_uint8_rgb(image)
    overlay = np.zeros_like(image_rgb)

    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    true_positive = pred_mask & gt_mask
    false_positive = pred_mask & (~gt_mask)
    false_negative = (~pred_mask) & gt_mask

    overlay[true_positive] = np.array([0, 255, 0], dtype=np.uint8)
    overlay[false_positive] = np.array([0, 0, 255], dtype=np.uint8)
    overlay[false_negative] = np.array([255, 0, 0], dtype=np.uint8)

    blended = np.clip(0.45 * image_rgb.astype(np.float32) + 0.55 * overlay.astype(np.float32), 0, 255)
    return blended.astype(np.uint8)


def save_prediction_visualization(image, gt_mask, pred_mask, epoch_dir, tile_dir, tile_id, epoch_index):
    left = _to_uint8_rgb(image)
    right = _build_confusion_overlay(image=image, pred_mask=pred_mask, gt_mask=gt_mask)
    canvas = _combine_side_by_side(left, right, separator_color=(255, 255, 0))

    epoch_filename = "%s_epoch_%03d.png" % (tile_id, epoch_index)
    tile_filename = "epoch_%03d.png" % epoch_index
    _save_image(canvas, os.path.join(epoch_dir, epoch_filename))
    _save_image(canvas, os.path.join(tile_dir, tile_filename))


def parse_args():
    parser = argparse.ArgumentParser(description="Helper CLI for visualization utilities.")
    parser.add_argument("--image-path", help="Optional image path.")
    parser.add_argument("--mask-path", help="Optional mask path.")
    parser.add_argument("--output-path", help="Optional output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.image_path and args.mask_path and args.output_path:
        image = load_tif_image(args.image_path)
        mask = load_tif_mask(args.mask_path)
        canvas = _combine_side_by_side(_to_uint8_rgb(image), _mask_to_rgb(mask))
        _save_image(canvas, args.output_path)


if __name__ == "__main__":
    main()
