import argparse
import csv
import os
import sys

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
from utils import cleanup_distributed, ensure_dir, is_main_process, save_json, set_seed, timestamped_run_id
from utils.distributed import find_free_port, init_distributed, reduce_sum_tensor
from visualization import save_augmentation_examples, save_prediction_visualization


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
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-train-viz", "--n_train_viz", dest="n_train_viz", type=int, default=100)
    parser.add_argument("--n-test-viz", "--n_test_viz", dest="n_test_viz", type=int, default=5)
    parser.add_argument("--viz-augs", "--viz_augs", dest="viz_augs", action="store_true")
    parser.set_defaults(viz_augs=False)
    parser.add_argument("--poly-power", "--poly_power", dest="poly_power", type=float, default=0.9)
    parser.add_argument("--min-lr", "--min_lr", dest="min_lr", type=float, default=1e-6)
    parser.add_argument("--save-every", "--save_every", dest="save_every", type=int, default=1)
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
    if len(args.image_size) != 2:
        raise ValueError("image_size must contain exactly two values: height width.")
    if args.mode == "test" and not args.checkpoint:
        raise ValueError("--mode test requires --checkpoint.")
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
        torch.distributed.barrier()


def resolve_run_directory(args):
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
    train_dataset = RoofDataset(args.train_test_dataset_path, split="train", transforms=train_transforms)
    test_dataset = RoofDataset(args.train_test_dataset_path, split="test", transforms=eval_transforms)
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


def create_model(args, device, distributed, rank):
    model = UNetPlusPlus(in_channels=3, out_channels=1, base_channels=args.base_channels)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    return model


def append_history_row(history_path, row):
    ensure_dir(os.path.dirname(history_path))
    fieldnames = [
        "epoch",
        "train_loss",
        "test_loss",
        "precision",
        "recall",
        "f1_score",
        "iou",
        "lr",
    ]
    file_exists = os.path.exists(history_path)
    with open(history_path, "a") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(checkpoint_path, epoch, model, optimizer, scheduler, scaler, args, metrics, best_f1):
    payload = {
        "epoch": epoch,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "args": namespace_to_dict(args),
        "metrics": metrics,
        "best_f1": best_f1,
    }
    ensure_dir(os.path.dirname(checkpoint_path))
    torch.save(payload, checkpoint_path)


def reduce_loss_sum(loss_sum, sample_count, device):
    state = torch.tensor([loss_sum, sample_count], dtype=torch.float64, device=device)
    state = reduce_sum_tensor(state)
    reduced_loss_sum, reduced_sample_count = [float(item) for item in state.tolist()]
    return reduced_loss_sum, reduced_sample_count


def train_one_epoch(model, loader, sampler, optimizer, scheduler, criterion, scaler, device, epoch, args):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    total_samples = 0.0
    progress = tqdm(
        loader,
        total=len(loader),
        leave=True,
        disable=not is_main_process(),
        desc="Epoch %03d/%03d [train]" % (epoch, args.n_epochs),
    )

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

        batch_size = float(images.size(0))
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        if is_main_process():
            progress.set_postfix(loss="%.4f" % (total_loss / max(total_samples, 1.0)))

    total_loss, total_samples = reduce_loss_sum(total_loss, total_samples, device)
    return total_loss / max(total_samples, 1.0)


def evaluate(model, loader, criterion, device, epoch, args):
    model.eval()
    meter = SegmentationMeter(threshold=args.threshold)
    total_loss = 0.0
    total_samples = 0.0

    progress = tqdm(
        loader,
        total=len(loader),
        leave=True,
        disable=not is_main_process(),
        desc="Epoch %03d/%03d [test]" % (epoch, args.n_epochs if args.mode == "train" else 1),
    )

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
            if is_main_process():
                progress.set_postfix(loss="%.4f" % (total_loss / max(total_samples, 1.0)))

    total_loss, total_samples = reduce_loss_sum(total_loss, total_samples, device)
    meter.synchronize_between_processes(device)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(total_samples, 1.0)
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
    progress = tqdm(
        loader,
        total=len(loader),
        leave=True,
        disable=not is_main_process(),
        desc="Epoch %03d [viz %s]" % (epoch_index, split),
    )
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
    history_path = os.path.join(run_dir, "history.csv")
    config_path = os.path.join(run_dir, "config.json")

    if rank == 0 and args.mode == "train":
        ensure_dir(run_dir)
        ensure_dir(checkpoints_dir)
        save_json(namespace_to_dict(args), config_path)

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

    model = create_model(args, device, distributed, rank)
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
        checkpoint = torch.load(args.checkpoint, map_location=device)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        metrics = evaluate(model, test_loader, criterion, device, epoch=checkpoint.get("epoch", 0), args=args)
        if rank == 0:
            save_json(metrics, os.path.join(run_dir, "test_metrics.json"))
            visualize_split(
                model=unwrap_model(model),
                train_test_dataset_path=args.train_test_dataset_path,
                split="test",
                records=selected_test_records,
                eval_transforms=eval_transforms,
                run_dir=run_dir,
                epoch_index=int(checkpoint.get("epoch", 0)),
                device=device,
                threshold=args.threshold,
            )
        if distributed:
            torch.distributed.barrier()
        if distributed:
            cleanup_distributed()
        return

    best_f1 = -1.0
    for epoch in range(1, args.n_epochs + 1):
        train_loss = train_one_epoch(
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
        )

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_metrics["loss"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1_score": test_metrics["f1_score"],
                "iou": test_metrics["iou"],
                "lr": current_lr,
            }
            append_history_row(history_path, history_row)

            if test_metrics["f1_score"] > best_f1:
                best_f1 = test_metrics["f1_score"]
                save_checkpoint(
                    checkpoint_path=os.path.join(checkpoints_dir, "best_f1.pth"),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    metrics=test_metrics,
                    best_f1=best_f1,
                )

            if epoch % args.save_every == 0:
                save_checkpoint(
                    checkpoint_path=os.path.join(checkpoints_dir, "last.pth"),
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    metrics=test_metrics,
                    best_f1=best_f1,
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
            torch.distributed.barrier()

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
