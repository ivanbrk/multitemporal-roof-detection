import argparse
import copy
import os
import sys
import time

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from augmentations import build_eval_transforms, build_train_transforms
from dataset import RoofDataset
from evaluation import SegmentationMeter
from losses import build_loss
from models import UNetPlusPlus
from preprocessing import build_train_test_dataset
from schedulers import build_scheduler
from utils import cleanup_distributed, distributed_barrier, ensure_dir, is_main_process, save_json, save_yaml, set_seed, timestamped_run_id
from utils.distributed import find_free_port, init_distributed, reduce_sum_tensor
from visualization import save_augmentation_examples, save_prediction_visualization


EVALUATION_COLUMNS = [
    "epoch",
    "epoch duration sec",
    "lr",
    "train loss",
    "train prec",
    "train rec",
    "train f1",
    "train iou",
    "test loss",
    "test prec",
    "test rec",
    "test f1",
    "test iou",
    "best f1",
    "best f1 epoch",
    "best iou",
    "best iou epoch",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate the roofs_2026 segmentation model.")
    parser.add_argument("--mode", choices=["train", "preprocess", "test"], default="train")
    parser.add_argument("--image-path", "--image_path", dest="image_path", required=True)
    parser.add_argument("--mask-path", "--mask_path", dest="mask_path", required=True)
    parser.add_argument(
        "--train-test-dataset-path",
        "--train_test_dataset_path",
        dest="train_test_dataset_path",
        default=os.path.join(CURRENT_DIR, "dataset", "train_test_dataset.xlsx"),
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        default=os.path.join(CURRENT_DIR, "runs", "training"),
    )
    parser.add_argument("--run-id", "--run_id", dest="run_id", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--refresh-train-test-dataset",
        "--refresh_train_test_dataset",
        dest="refresh_train_test_dataset",
        action="store_true",
    )
    parser.add_argument("--test-size", "--test_size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pixel-size-m", "--pixel_size_m", dest="pixel_size_m", type=float, default=0.5)
    parser.add_argument("--n-epochs", "--n_epochs", dest="n_epochs", type=int, default=100)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", default="tversky_focal")
    parser.add_argument("--scheduler", default="poly")
    parser.add_argument(
        "--loss-weights",
        "--loss_weights",
        "--weights",
        dest="loss_weights",
        nargs=2,
        type=float,
        default=None,
        metavar=("BG_WEIGHT", "FG_WEIGHT"),
    )
    parser.add_argument("--n-devices", "--n_devices", dest="n_devices", type=int, default=1)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--base-channels", "--base_channels", dest="base_channels", type=int, default=24)
    parser.add_argument("--image-size", "--image_size", dest="image_size", nargs=2, type=int, default=[1000, 1000])
    parser.add_argument("--train-limit", "--train_limit", dest="train_limit", type=int, default=None)
    parser.add_argument("--test-limit", "--test_limit", dest="test_limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-train-viz", "--n_train_viz", dest="n_train_viz", type=int, default=100)
    parser.add_argument("--n-test-viz", "--n_test_viz", dest="n_test_viz", type=int, default=5)
    parser.add_argument("--viz-augs", "--viz_augs", dest="viz_augs", action="store_true")
    parser.set_defaults(viz_augs=False)
    parser.add_argument("--poly-power", "--poly_power", dest="poly_power", type=float, default=0.9)
    parser.add_argument("--min-lr", "--min_lr", dest="min_lr", type=float, default=1e-6)
    return parser.parse_args()


def namespace_to_dict(args):
    serializable = {}
    for key, value in vars(args).items():
        if isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return serializable


def validate_args(args):
    if args.n_devices < 1:
        raise ValueError("n_devices must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if args.n_epochs < 1 and args.mode == "train":
        raise ValueError("n_epochs must be >= 1 for training.")
    if args.train_limit is not None and args.train_limit < 1:
        raise ValueError("train_limit must be >= 1 when provided.")
    if args.test_limit is not None and args.test_limit < 1:
        raise ValueError("test_limit must be >= 1 when provided.")
    if len(args.image_size) != 2:
        raise ValueError("image_size must contain exactly two values: height width.")
    if args.mode == "test" and not args.checkpoint:
        raise ValueError("--mode test requires --checkpoint.")
    if args.resume and args.mode != "train":
        raise ValueError("--resume can be used only with --mode train.")
    if args.resume and not os.path.isfile(args.resume):
        raise ValueError("Resume checkpoint not found: %s" % args.resume)
    if args.resume and args.run_id is not None:
        raise ValueError("--run-id cannot be used together with --resume. Resume continues the original run directory.")
    if args.n_devices > 1 and not torch.cuda.is_available():
        raise RuntimeError("DDP is supported only when CUDA is available.")
    if torch.cuda.is_available() and args.n_devices > torch.cuda.device_count():
        raise RuntimeError(
            "Requested %s GPUs, but only %s are available." % (args.n_devices, torch.cuda.device_count())
        )


def ensure_train_test_dataset_ready(args, rank, distributed):
    train_test_dataset_dir = os.path.dirname(args.train_test_dataset_path)
    if train_test_dataset_dir:
        ensure_dir(train_test_dataset_dir)

    should_build = args.refresh_train_test_dataset or (not os.path.exists(args.train_test_dataset_path))
    if rank == 0 and should_build:
        build_train_test_dataset(
            image_path=args.image_path,
            mask_path=args.mask_path,
            output_xlsx=args.train_test_dataset_path,
            test_size=args.test_size,
            seed=args.seed,
            pixel_size_m=args.pixel_size_m,
        )
    if distributed:
        distributed_barrier()


def resolve_run_directory(args):
    if args.mode == "train" and args.resume:
        checkpoints_dir = os.path.dirname(os.path.abspath(args.resume))
        return os.path.dirname(checkpoints_dir)
    if args.mode == "test" and args.checkpoint:
        checkpoints_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        return os.path.dirname(checkpoints_dir)
    run_id = args.run_id or timestamped_run_id("train")
    return os.path.join(args.output_dir, run_id)


def create_dataloader(dataset, batch_size, shuffle, num_workers, distributed):
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=bool(num_workers > 0),
    )
    return loader, sampler


def build_datasets(args):
    train_transforms = build_train_transforms(tuple(args.image_size))
    eval_transforms = build_eval_transforms(tuple(args.image_size))
    train_dataset = RoofDataset(
        args.train_test_dataset_path,
        split="train",
        transforms=train_transforms,
        limit=args.train_limit,
    )
    test_dataset = RoofDataset(
        args.train_test_dataset_path,
        split="test",
        transforms=eval_transforms,
        limit=args.test_limit,
    )
    return train_dataset, test_dataset, train_transforms, eval_transforms


def get_selected_records(train_test_dataset_path, split, n_samples, seed):
    if n_samples <= 0:
        return []
    dataframe = pd.read_excel(train_test_dataset_path)
    dataframe = dataframe[dataframe["train_test"] == split].copy()
    if dataframe.empty:
        return []
    dataframe = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return dataframe.head(int(n_samples)).to_dict("records")


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def use_progress_bar():
    return is_main_process() and sys.stderr.isatty()


def create_progress(iterable, total, desc):
    if use_progress_bar():
        return (
            tqdm(
                iterable,
                total=total,
                leave=True,
                disable=False,
                desc=desc,
            ),
            True,
        )
    return iterable, False


def log_phase_completion(desc, message):
    if is_main_process():
        print("%s: %s" % (desc, message), flush=True)


def format_duration(seconds):
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return "%02d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)


def make_serializable(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): make_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_model_config(args):
    return {
        "name": "unetplusplus",
        "class_name": "UNetPlusPlus",
        "in_channels": 3,
        "out_channels": 1,
        "base_channels": int(args.base_channels),
        "image_size": [int(args.image_size[0]), int(args.image_size[1])],
    }


def build_loss_config(args, criterion):
    config = {
        "name": str(args.loss),
        "class_name": criterion.__class__.__name__,
        "class_weights": list(args.loss_weights) if args.loss_weights is not None else None,
    }
    for attribute in ("alpha", "beta", "gamma", "smooth", "bg_weight", "fg_weight"):
        if hasattr(criterion, attribute):
            config[attribute] = make_serializable(getattr(criterion, attribute))
    if hasattr(criterion, "pos_weight"):
        config["pos_weight"] = make_serializable(criterion.pos_weight)
    return config


def build_optimizer_config(optimizer):
    return {
        "name": optimizer.__class__.__name__,
        "defaults": make_serializable(optimizer.defaults),
    }


def build_scheduler_config(args, scheduler, total_steps):
    config = {
        "name": str(args.scheduler),
        "class_name": scheduler.__class__.__name__,
        "total_steps": int(total_steps),
        "power": float(args.poly_power),
        "min_lr": float(args.min_lr),
    }
    if hasattr(scheduler, "max_steps"):
        config["max_steps"] = int(scheduler.max_steps)
    return config


def build_run_config(args, model_config, optimizer_config, scheduler_config, loss_config, run_dir, checkpoints_dir, start_epoch):
    return {
        "project": "roofs_2026",
        "run": {
            "run_id": os.path.basename(run_dir),
            "mode": args.mode,
            "start_epoch": int(start_epoch),
            "target_epochs": int(args.n_epochs),
            "n_devices": int(args.n_devices),
            "per_device_batch_size": int(args.batch_size),
            "global_batch_size": int(args.batch_size) * int(args.n_devices),
            "num_workers_per_process": int(args.num_workers),
            "seed": int(args.seed),
            "threshold": float(args.threshold),
            "resume": args.resume,
        },
        "paths": {
            "image_path": args.image_path,
            "mask_path": args.mask_path,
            "train_test_dataset_path": args.train_test_dataset_path,
            "output_dir": args.output_dir,
            "run_dir": run_dir,
            "checkpoints_dir": checkpoints_dir,
            "resume_checkpoint": args.resume,
            "test_checkpoint": args.checkpoint,
        },
        "dataset": {
            "image_size": [int(args.image_size[0]), int(args.image_size[1])],
            "pixel_size_m": float(args.pixel_size_m),
            "test_size": float(args.test_size),
            "train_limit": args.train_limit,
            "test_limit": args.test_limit,
        },
        "model": model_config,
        "optimizer": optimizer_config,
        "scheduler": scheduler_config,
        "loss": loss_config,
        "training": {
            "viz_augs": bool(args.viz_augs),
            "n_train_viz": int(args.n_train_viz),
            "n_test_viz": int(args.n_test_viz),
            "cli_arguments": namespace_to_dict(args),
        },
    }


def save_evaluation_rows(evaluation_path, rows):
    ensure_dir(os.path.dirname(evaluation_path))
    dataframe = pd.DataFrame(rows, columns=EVALUATION_COLUMNS)
    dataframe.to_excel(evaluation_path, index=False)


def load_evaluation_rows(evaluation_path):
    if not os.path.exists(evaluation_path):
        return []
    dataframe = pd.read_excel(evaluation_path)
    for column in EVALUATION_COLUMNS:
        if column not in dataframe.columns:
            dataframe[column] = None
    dataframe = dataframe[EVALUATION_COLUMNS]
    return dataframe.to_dict("records")


def build_epoch_summary(
    epoch,
    total_epochs,
    train_metrics,
    test_metrics,
    epoch_seconds,
    best_f1,
    best_f1_epoch,
    best_iou,
    best_iou_epoch,
):
    return (
        "Epoch %03d/%03d | train_loss=%.4f | test_loss=%.4f | "
        "train_prec=%.4f | train_rec=%.4f | train_f1=%.4f | train_iou=%.4f | "
        "test_prec=%.4f | test_rec=%.4f | test_f1=%.4f | test_iou=%.4f | "
        "duration=%s | best_test_f1=%.4f (epoch %03d) | best_test_iou=%.4f (epoch %03d)"
        % (
            epoch,
            total_epochs,
            train_metrics["loss"],
            test_metrics["loss"],
            train_metrics["precision"],
            train_metrics["recall"],
            train_metrics["f1_score"],
            train_metrics["iou"],
            test_metrics["precision"],
            test_metrics["recall"],
            test_metrics["f1_score"],
            test_metrics["iou"],
            format_duration(epoch_seconds),
            best_f1,
            best_f1_epoch,
            best_iou,
            best_iou_epoch,
        )
    )


def load_checkpoint_payload(checkpoint_path):
    return torch.load(checkpoint_path, map_location="cpu")


def create_model(model_config, device, distributed, rank):
    model = UNetPlusPlus(
        in_channels=int(model_config["in_channels"]),
        out_channels=int(model_config["out_channels"]),
        base_channels=int(model_config["base_channels"]),
    )
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    return model


def save_checkpoint(
    checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    run_config,
    model_config,
    metrics,
    best_f1,
    best_f1_epoch,
    best_iou,
    best_iou_epoch,
    best_f1_checkpoint_path,
    best_iou_checkpoint_path,
):
    full_model = copy.deepcopy(unwrap_model(model)).cpu()
    full_model.eval()
    payload = {
        "epoch": epoch,
        "model": full_model,
        "model_state": unwrap_model(model).state_dict(),
        "model_config": make_serializable(model_config),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "args": namespace_to_dict(args),
        "run_config": make_serializable(run_config),
        "metrics": metrics,
        "best_f1": best_f1,
        "best_f1_epoch": best_f1_epoch,
        "best_iou": best_iou,
        "best_iou_epoch": best_iou_epoch,
        "best_f1_checkpoint_path": best_f1_checkpoint_path,
        "best_iou_checkpoint_path": best_iou_checkpoint_path,
    }
    ensure_dir(os.path.dirname(checkpoint_path))
    torch.save(payload, checkpoint_path)


def update_best_checkpoint(
    checkpoints_dir,
    checkpoint_prefix,
    previous_checkpoint_path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    run_config,
    model_config,
    metrics,
    best_f1,
    best_f1_epoch,
    best_iou,
    best_iou_epoch,
    best_f1_checkpoint_path,
    best_iou_checkpoint_path,
):
    new_checkpoint_path = os.path.join(checkpoints_dir, "%s_%03d.pt" % (checkpoint_prefix, epoch))
    if previous_checkpoint_path and previous_checkpoint_path != new_checkpoint_path and os.path.exists(previous_checkpoint_path):
        os.remove(previous_checkpoint_path)
    save_checkpoint(
        checkpoint_path=new_checkpoint_path,
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        args=args,
        run_config=run_config,
        model_config=model_config,
        metrics=metrics,
        best_f1=best_f1,
        best_f1_epoch=best_f1_epoch,
        best_iou=best_iou,
        best_iou_epoch=best_iou_epoch,
        best_f1_checkpoint_path=best_f1_checkpoint_path,
        best_iou_checkpoint_path=best_iou_checkpoint_path,
    )
    return new_checkpoint_path


def cleanup_checkpoint_directory(checkpoints_dir, keep_filenames=None):
    if not os.path.isdir(checkpoints_dir):
        return
    keep_filenames = set(keep_filenames or [])
    removable_names = {"best_f1.pth", "best_iou.pth", "last.pth"}
    for filename in os.listdir(checkpoints_dir):
        path = os.path.join(checkpoints_dir, filename)
        if not os.path.isfile(path):
            continue
        if filename in keep_filenames:
            continue
        if filename in removable_names or (filename.startswith("best_") and filename.endswith(".pt")):
            os.remove(path)


def checkpoint_filename_from_epoch(prefix, epoch):
    if epoch is None or int(epoch) <= 0:
        return None
    return "%s_%03d.pt" % (prefix, int(epoch))


def reduce_loss_sum(loss_sum, sample_count, device):
    state = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
    state = reduce_sum_tensor(state)
    reduced_loss_sum, reduced_sample_count = [float(item) for item in state.tolist()]
    return reduced_loss_sum, reduced_sample_count


def train_one_epoch(model, loader, sampler, optimizer, scheduler, criterion, scaler, device, epoch, args):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    meter = SegmentationMeter(threshold=args.threshold)
    total_loss = 0.0
    total_samples = 0.0
    desc = "Epoch %03d/%03d [train]" % (epoch, args.n_epochs)
    progress, use_tqdm = create_progress(loader, total=len(loader), desc=desc)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        meter.update_from_logits(logits.detach(), masks)
        batch_size = float(images.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        if use_tqdm:
            progress.set_postfix(loss="%.4f" % (total_loss / max(total_samples, 1.0)))

    total_loss, total_samples = reduce_loss_sum(total_loss, total_samples, device)
    meter.synchronize_between_processes(device)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(total_samples, 1.0)
    return metrics


def evaluate(model, loader, criterion, device, epoch, args, log_completion=None):
    model.eval()
    meter = SegmentationMeter(threshold=args.threshold)
    total_loss = 0.0
    total_samples = 0.0

    desc = "Epoch %03d/%03d [test]" % (epoch, args.n_epochs if args.mode == "train" else 1)
    progress, use_tqdm = create_progress(loader, total=len(loader), desc=desc)

    with torch.no_grad():
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, masks)
            meter.update_from_logits(logits, masks)
            batch_size = float(images.size(0))
            total_loss += float(loss.detach().item()) * batch_size
            total_samples += batch_size
            if use_tqdm:
                progress.set_postfix(loss="%.4f" % (total_loss / max(total_samples, 1.0)))

    total_loss, total_samples = reduce_loss_sum(total_loss, total_samples, device)
    meter.synchronize_between_processes(device)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(total_samples, 1.0)
    if log_completion is None:
        log_completion = args.mode == "test"
    if (not use_tqdm) and log_completion:
        log_phase_completion(
            desc,
            "completed with loss=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, iou=%.4f."
            % (
                metrics["loss"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["iou"],
            ),
        )
    return metrics


def visualize_split(
    model,
    train_test_dataset_path,
    split,
    records,
    eval_transforms,
    run_dir,
    epoch_index,
    device,
    threshold,
):
    if not records:
        return

    split_dir = os.path.join(run_dir, "visualize", split)
    epoch_dir = os.path.join(split_dir, "per_epoch", "epoch_%03d" % epoch_index)
    per_tile_root = os.path.join(split_dir, "per_tile")
    ensure_dir(epoch_dir)
    ensure_dir(per_tile_root)

    selected_tiles = [record["tile_id"] for record in records]
    dataset = RoofDataset(
        train_test_dataset_path=train_test_dataset_path,
        split=split,
        transforms=eval_transforms,
        selected_tiles=selected_tiles,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()
    desc = "Epoch %03d [viz %s]" % (epoch_index, split)
    progress, use_tqdm = create_progress(loader, total=len(loader), desc=desc)
    saved_count = 0
    with torch.no_grad():
        for batch in progress:
            image_tensor = batch["image"].to(device, non_blocking=True)
            mask_tensor = batch["mask"].to(device, non_blocking=True)
            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)
            prediction = (probabilities >= threshold).float()

            tile_id = batch["tile_id"][0]
            image = image_tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
            gt_mask = mask_tensor[0, 0].detach().cpu().numpy()
            pred_mask = prediction[0, 0].detach().cpu().numpy()
            tile_dir = os.path.join(per_tile_root, tile_id)
            save_prediction_visualization(
                image=image,
                gt_mask=gt_mask,
                pred_mask=pred_mask,
                epoch_dir=epoch_dir,
                tile_dir=tile_dir,
                tile_id=tile_id,
                epoch_index=epoch_index,
            )
            saved_count += 1
    if not use_tqdm:
        log_phase_completion(desc, "saved %d visualization tiles." % saved_count)


def maybe_save_augmentation_visualizations(records, train_transforms, run_dir, args):
    if not args.viz_augs or not records:
        return
    output_dir = os.path.join(run_dir, "visualize", "augs")
    save_augmentation_examples(
        records=records,
        transform=train_transforms,
        output_dir=output_dir,
        max_examples=min(args.n_train_viz, len(records)),
    )


def run_training(rank, args):
    distributed = init_distributed(rank, args.n_devices, args.master_port) if args.n_devices > 1 else False
    device = torch.device("cuda:%d" % rank if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + rank)

    ensure_train_test_dataset_ready(args, rank=rank, distributed=distributed)

    run_dir = resolve_run_directory(args)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    evaluation_path = os.path.join(run_dir, "evaluation.xlsx")
    config_json_path = os.path.join(run_dir, "config.json")
    config_yaml_path = os.path.join(run_dir, "config.yaml")

    checkpoint_source = None
    if args.mode == "train" and args.resume:
        checkpoint_source = args.resume
    elif args.mode == "test" and args.checkpoint:
        checkpoint_source = args.checkpoint
    checkpoint_payload = load_checkpoint_payload(checkpoint_source) if checkpoint_source else None

    train_dataset, test_dataset, train_transforms, eval_transforms = build_datasets(args)
    train_loader, train_sampler = create_dataloader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        distributed=distributed,
    )
    test_loader, _ = create_dataloader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        distributed=distributed,
    )

    model_config = build_model_config(args)
    if checkpoint_payload is not None and checkpoint_payload.get("model_config"):
        model_config = make_serializable(checkpoint_payload["model_config"])

    model = create_model(model_config, device, distributed, rank)
    criterion = build_loss(args.loss, class_weights=args.loss_weights).to(device)
    optimizer = torch.optim.AdamW(unwrap_model(model).parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.n_epochs)
    scheduler = build_scheduler(
        name=args.scheduler,
        optimizer=optimizer,
        total_steps=total_steps,
        power=args.poly_power,
        min_lr=args.min_lr,
    )
    scaler = GradScaler(enabled=torch.cuda.is_available())

    start_epoch = 1
    best_f1 = -1.0
    best_f1_epoch = 0
    best_iou = -1.0
    best_iou_epoch = 0
    best_f1_checkpoint_path = None
    best_iou_checkpoint_path = None
    evaluation_rows = []

    if checkpoint_payload is not None:
        unwrap_model(model).load_state_dict(checkpoint_payload["model_state"])

    if args.mode == "train" and checkpoint_payload is not None:
        if checkpoint_payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(checkpoint_payload["optimizer_state"])
        if checkpoint_payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint_payload["scheduler_state"])
        if checkpoint_payload.get("scaler_state") is not None:
            scaler.load_state_dict(checkpoint_payload["scaler_state"])
        if hasattr(scheduler, "max_steps"):
            scheduler.max_steps = total_steps
        if hasattr(scheduler, "power"):
            scheduler.power = float(args.poly_power)
        if hasattr(scheduler, "min_lr"):
            scheduler.min_lr = float(args.min_lr)

        start_epoch = int(checkpoint_payload.get("epoch", 0)) + 1
        best_f1 = float(checkpoint_payload.get("best_f1", -1.0))
        best_f1_epoch = int(checkpoint_payload.get("best_f1_epoch", 0))
        best_iou = float(checkpoint_payload.get("best_iou", -1.0))
        best_iou_epoch = int(checkpoint_payload.get("best_iou_epoch", 0))
        evaluation_rows = load_evaluation_rows(evaluation_path)

        stored_best_f1_path = checkpoint_payload.get("best_f1_checkpoint_path")
        stored_best_iou_path = checkpoint_payload.get("best_iou_checkpoint_path")
        if stored_best_f1_path:
            best_f1_checkpoint_path = (
                stored_best_f1_path
                if os.path.isabs(stored_best_f1_path)
                else os.path.join(checkpoints_dir, os.path.basename(stored_best_f1_path))
            )
        if stored_best_iou_path:
            best_iou_checkpoint_path = (
                stored_best_iou_path
                if os.path.isabs(stored_best_iou_path)
                else os.path.join(checkpoints_dir, os.path.basename(stored_best_iou_path))
            )
        if best_f1_checkpoint_path is None:
            filename = checkpoint_filename_from_epoch("best_f1", best_f1_epoch)
            if filename is not None:
                best_f1_checkpoint_path = os.path.join(checkpoints_dir, filename)
        if best_iou_checkpoint_path is None:
            filename = checkpoint_filename_from_epoch("best_iou", best_iou_epoch)
            if filename is not None:
                best_iou_checkpoint_path = os.path.join(checkpoints_dir, filename)

        if start_epoch > args.n_epochs:
            raise ValueError(
                "Resume checkpoint epoch %d already meets or exceeds target n_epochs=%d."
                % (start_epoch - 1, args.n_epochs)
            )

    optimizer_config = build_optimizer_config(optimizer)
    scheduler_config = build_scheduler_config(args, scheduler, total_steps)
    loss_config = build_loss_config(args, criterion)
    run_config = build_run_config(
        args=args,
        model_config=model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        loss_config=loss_config,
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        start_epoch=start_epoch,
    )

    if rank == 0 and args.mode == "train":
        ensure_dir(run_dir)
        ensure_dir(checkpoints_dir)
        keep_filenames = []
        if args.resume:
            if best_f1_checkpoint_path:
                keep_filenames.append(os.path.basename(best_f1_checkpoint_path))
            if best_iou_checkpoint_path:
                keep_filenames.append(os.path.basename(best_iou_checkpoint_path))
        cleanup_checkpoint_directory(checkpoints_dir, keep_filenames=keep_filenames)
        save_json(run_config, config_json_path)
        save_yaml(run_config, config_yaml_path)

    selected_train_records = []
    selected_test_records = []
    if rank == 0:
        selected_train_records = get_selected_records(
            args.train_test_dataset_path,
            split="train",
            n_samples=args.n_train_viz,
            seed=args.seed,
        )
        selected_test_records = get_selected_records(
            args.train_test_dataset_path,
            split="test",
            n_samples=args.n_test_viz,
            seed=args.seed,
        )
        maybe_save_augmentation_visualizations(selected_train_records, train_transforms, run_dir, args)

    if args.mode == "test":
        metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            epoch=checkpoint_payload.get("epoch", 0),
            args=args,
            log_completion=True,
        )
        if rank == 0:
            save_json(metrics, os.path.join(run_dir, "test_metrics.json"))
            visualize_split(
                model=unwrap_model(model),
                train_test_dataset_path=args.train_test_dataset_path,
                split="test",
                records=selected_test_records,
                eval_transforms=eval_transforms,
                run_dir=run_dir,
                epoch_index=int(checkpoint_payload.get("epoch", 0)),
                device=device,
                threshold=args.threshold,
            )
        if distributed:
            distributed_barrier()
        if distributed:
            cleanup_distributed()
        return

    for epoch in range(start_epoch, args.n_epochs + 1):
        epoch_start_time = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
        )
        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            args=args,
            log_completion=False,
        )
        epoch_duration = time.perf_counter() - epoch_start_time

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            new_best_f1 = test_metrics["f1_score"] > best_f1
            new_best_iou = test_metrics["iou"] > best_iou

            if new_best_f1:
                best_f1 = test_metrics["f1_score"]
                best_f1_epoch = epoch
            if new_best_iou:
                best_iou = test_metrics["iou"]
                best_iou_epoch = epoch

            next_best_f1_checkpoint_path = best_f1_checkpoint_path
            next_best_iou_checkpoint_path = best_iou_checkpoint_path
            if new_best_f1:
                next_best_f1_checkpoint_path = os.path.join(checkpoints_dir, "best_f1_%03d.pt" % epoch)
            if new_best_iou:
                next_best_iou_checkpoint_path = os.path.join(checkpoints_dir, "best_iou_%03d.pt" % epoch)

            if new_best_f1:
                best_f1_checkpoint_path = update_best_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    checkpoint_prefix="best_f1",
                    previous_checkpoint_path=best_f1_checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    run_config=run_config,
                    model_config=model_config,
                    metrics=test_metrics,
                    best_f1=best_f1,
                    best_f1_epoch=best_f1_epoch,
                    best_iou=best_iou,
                    best_iou_epoch=best_iou_epoch,
                    best_f1_checkpoint_path=next_best_f1_checkpoint_path,
                    best_iou_checkpoint_path=next_best_iou_checkpoint_path,
                )

            if new_best_iou:
                best_iou_checkpoint_path = update_best_checkpoint(
                    checkpoints_dir=checkpoints_dir,
                    checkpoint_prefix="best_iou",
                    previous_checkpoint_path=best_iou_checkpoint_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    run_config=run_config,
                    model_config=model_config,
                    metrics=test_metrics,
                    best_f1=best_f1,
                    best_f1_epoch=best_f1_epoch,
                    best_iou=best_iou,
                    best_iou_epoch=best_iou_epoch,
                    best_f1_checkpoint_path=best_f1_checkpoint_path or next_best_f1_checkpoint_path,
                    best_iou_checkpoint_path=next_best_iou_checkpoint_path,
                )

            evaluation_row = {
                "epoch": epoch,
                "epoch duration sec": epoch_duration,
                "lr": current_lr,
                "train loss": train_metrics["loss"],
                "train prec": train_metrics["precision"],
                "train rec": train_metrics["recall"],
                "train f1": train_metrics["f1_score"],
                "train iou": train_metrics["iou"],
                "test loss": test_metrics["loss"],
                "test prec": test_metrics["precision"],
                "test rec": test_metrics["recall"],
                "test f1": test_metrics["f1_score"],
                "test iou": test_metrics["iou"],
                "best f1": best_f1,
                "best f1 epoch": best_f1_epoch,
                "best iou": best_iou,
                "best iou epoch": best_iou_epoch,
            }
            evaluation_rows.append(evaluation_row)
            save_evaluation_rows(evaluation_path, evaluation_rows)
            print(
                build_epoch_summary(
                    epoch=epoch,
                    total_epochs=args.n_epochs,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    epoch_seconds=epoch_duration,
                    best_f1=best_f1,
                    best_f1_epoch=best_f1_epoch,
                    best_iou=best_iou,
                    best_iou_epoch=best_iou_epoch,
                ),
                flush=True,
            )

            visualize_split(
                model=unwrap_model(model),
                train_test_dataset_path=args.train_test_dataset_path,
                split="train",
                records=selected_train_records,
                eval_transforms=eval_transforms,
                run_dir=run_dir,
                epoch_index=epoch,
                device=device,
                threshold=args.threshold,
            )
            visualize_split(
                model=unwrap_model(model),
                train_test_dataset_path=args.train_test_dataset_path,
                split="test",
                records=selected_test_records,
                eval_transforms=eval_transforms,
                run_dir=run_dir,
                epoch_index=epoch,
                device=device,
                threshold=args.threshold,
            )
        if distributed:
            distributed_barrier()

    if distributed:
        cleanup_distributed()


if __name__ == "__main__":
    arguments = parse_args()
    validate_args(arguments)

    if arguments.mode == "preprocess":
        ensure_dir(os.path.dirname(arguments.train_test_dataset_path))
        build_train_test_dataset(
            image_path=arguments.image_path,
            mask_path=arguments.mask_path,
            output_xlsx=arguments.train_test_dataset_path,
            test_size=arguments.test_size,
            seed=arguments.seed,
            pixel_size_m=arguments.pixel_size_m,
        )
    else:
        arguments.master_port = find_free_port()
        if arguments.n_devices > 1:
            mp.spawn(run_training, nprocs=arguments.n_devices, args=(arguments,))
        else:
            run_training(rank=0, args=arguments)
