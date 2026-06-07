import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, decollate_batch, list_data_collate
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    SpatialPadd
)
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast

from brain_mri_3d_tumor_segmentation.segmentation.assignment_3d_unet.data import (
    ConvertBratsLabelToRegionsd,
    REGION_NAMES,
    load_decathlon_cases,
    split_cases,
    to_monai_records
)


SEGMENTATION_ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SEGMENTATION_ROOT / "decathlon"
DEFAULT_OUTPUT_DIR = ASSIGNMENT_MODEL_DIR / "checkpoints" / "assignment_unet"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the assignment baseline 3D U-Net for BRATS segmentation.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-case-limit", type=int)
    parser.add_argument("--val-case-limit", type=int, default=16)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--cache-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, args.deterministic)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_count, val_count = create_data_loaders(args)
    model = create_model().to(device)
    criterion = create_loss().to(device)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    post_prediction = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(device.type, enabled=use_amp)
    best_mean_dice = -1.0
    history = []

    save_training_config(output_dir, args, train_count, val_count, device)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        scheduler.step()
        should_validate = should_validate_epoch(epoch, args.epochs, args.val_interval)
        epoch_metrics = {
            "epoch": epoch,
            "trainLoss": train_loss,
            "learningRate": scheduler.get_last_lr()[0]
        }
        val_metrics = {
            "validated": False
        }
        is_best = False

        if should_validate:
            val_metrics = validate(model, val_loader, criterion, dice_metric, post_prediction, device, args)
            mean_dice = val_metrics["meanDice"]
            is_best = mean_dice > best_mean_dice

            if is_best:
                best_mean_dice = mean_dice

            epoch_metrics.update(val_metrics)

        history.append(epoch_metrics)
        write_json(output_dir / "history.json", history)

        save_checkpoint(output_dir / "latest_model.pth", model, optimizer, scheduler, args, epoch, val_metrics)

        if is_best:
            save_checkpoint(output_dir / "best_metric_model.pth", model, optimizer, scheduler, args, epoch, val_metrics)

        print_training_progress(epoch, args.epochs, train_loss, val_metrics, best_mean_dice)


def should_validate_epoch(epoch, total_epochs, val_interval):
    interval = max(1, val_interval)

    return epoch == 1 or epoch % interval == 0 or epoch == total_epochs


def print_training_progress(epoch, total_epochs, train_loss, val_metrics, best_mean_dice):
    if not val_metrics.get("validated", True):
        print(
            f"epoch={epoch:03d}/{total_epochs:03d} "
            f"train_loss={train_loss:.5f} "
            f"validation=skipped "
            f"best={best_mean_dice:.4f}",
            flush=True
        )
        return

    print(
        f"epoch={epoch:03d}/{total_epochs:03d} "
        f"train_loss={train_loss:.5f} "
        f"val_loss={val_metrics['valLoss']:.5f} "
        f"dice_tc={val_metrics['diceTC']:.4f} "
        f"dice_wt={val_metrics['diceWT']:.4f} "
        f"dice_et={val_metrics['diceET']:.4f} "
        f"mean_dice={val_metrics['meanDice']:.4f} "
        f"best={best_mean_dice:.4f}",
        flush=True
    )


def set_seed(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        set_determinism(seed=seed)
    else:
        torch.backends.cudnn.benchmark = True


def create_data_loaders(args):
    cases = load_decathlon_cases(args.data_dir)
    train_cases, val_cases = split_cases(cases, args.val_ratio, args.seed)
    selected_train_cases = train_cases
    selected_val_cases = val_cases[:args.val_case_limit]

    if args.train_case_limit:
        selected_train_cases = train_cases[:args.train_case_limit]

    train_records = to_monai_records(selected_train_cases)
    val_records = to_monai_records(selected_val_cases)
    use_persistent_workers = args.num_workers > 0
    train_dataset = CacheDataset(
        data=train_records,
        transform=create_train_transform(args),
        cache_rate=args.cache_rate,
        num_workers=args.num_workers
    )
    val_dataset = CacheDataset(
        data=val_records,
        transform=create_val_transform(args),
        cache_rate=args.cache_rate,
        num_workers=args.num_workers
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
        collate_fn=list_data_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )

    return train_loader, val_loader, len(train_records), len(val_records)


def create_train_transform(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image", channel_dim=-1),
        EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
        ConvertBratsLabelToRegionsd(keys="label"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=tuple(args.patch_size)),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=tuple(args.patch_size),
            pos=1,
            neg=1,
            num_samples=args.patches_per_case,
            image_key="image",
            image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3, spatial_axes=(0, 1)),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.15),
        EnsureTyped(keys=["image", "label"])
    ])


def create_val_transform(args):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image", channel_dim=-1),
        EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
        ConvertBratsLabelToRegionsd(keys="label"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=tuple(args.patch_size)),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ])


def create_model():
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE"
    )


def create_loss():
    return DiceFocalLoss(
        sigmoid=True,
        squared_pred=True,
        smooth_nr=0,
        smooth_dr=1e-5,
        batch=True,
        gamma=2.0,
        lambda_dice=1.0,
        lambda_focal=1.0
    )


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    losses = []

    for batch in train_loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(image)
            loss = criterion(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return float(np.mean(losses))


def validate(model, val_loader, criterion, dice_metric, post_prediction, device, args):
    model.eval()
    losses = []
    use_amp = device.type == "cuda" and not args.no_amp
    dice_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=tuple(args.patch_size),
                    sw_batch_size=args.sw_batch_size,
                    predictor=model,
                    overlap=args.overlap
                )
                loss = criterion(logits, label)

            predictions = [post_prediction(prediction) for prediction in decollate_batch(logits)]
            labels = decollate_batch(label)
            losses.append(loss.item())
            dice_metric(y_pred=predictions, y=labels)

    dice_by_region = dice_metric.aggregate().detach().cpu().numpy()
    dice_metric.reset()

    return {
        "valLoss": float(np.mean(losses)),
        "diceTC": float(dice_by_region[0]),
        "diceWT": float(dice_by_region[1]),
        "diceET": float(dice_by_region[2]),
        "meanDice": float(dice_by_region.mean())
    }


def save_training_config(output_dir, args, train_count, val_count, device):
    config = vars(args).copy()
    config["dataDir"] = str(config["data_dir"])
    config["outputDir"] = str(config["output_dir"])
    config["regionNames"] = REGION_NAMES
    config["trainCaseCount"] = train_count
    config["valCaseCount"] = val_count
    config["device"] = str(device)
    config.pop("data_dir")
    config.pop("output_dir")
    write_json(output_dir / "training_config.json", config)


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, args, epoch, val_metrics):
    checkpoint = {
        "modelName": "assignment_unet3d_baseline",
        "epoch": epoch,
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
        "schedulerState": scheduler.state_dict(),
        "args": vars(args),
        "regionNames": REGION_NAMES,
        "metrics": val_metrics
    }
    torch.save(checkpoint, checkpoint_path)


def write_json(file_path, content):
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
