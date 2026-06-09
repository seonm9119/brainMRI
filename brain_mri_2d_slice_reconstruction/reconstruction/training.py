import json
import random
from datetime import datetime, timezone

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from brain_mri_2d_slice_reconstruction.reconstruction.data import Brain2DSliceDataset
from brain_mri_2d_slice_reconstruction.reconstruction.early_stopping import create_early_stopping_monitor
from brain_mri_2d_slice_reconstruction.reconstruction.metrics import ReconstructionLoss, calculate_reconstruction_metrics


def run_training(args, output_dir, create_model, model_name, model_title):
    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_count, val_count = create_data_loaders(args)
    model = create_model().to(device)
    criterion = ReconstructionLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        gradient_weight=args.gradient_weight,
        foreground_weight=getattr(args, "foreground_weight", 0.0),
        background_weight=getattr(args, "background_weight", 0.25),
        foreground_threshold=getattr(args, "foreground_threshold", 0.02)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(device.type, enabled=use_amp)
    best_val_loss = float("inf")
    initialization = create_initialization_info()
    start_epoch = 1

    if getattr(args, "initialize_from", None):
        best_val_loss, initialization = load_model_weights(args.initialize_from, model, device, "explicit_checkpoint_finetune")
    elif (output_dir / "best_metric_model.pth").exists():
        best_val_loss, initialization = load_model_weights(output_dir / "best_metric_model.pth", model, device, "best_checkpoint_finetune")

    save_training_config(output_dir, args, model_name, model_title, train_count, val_count, device, initialization)
    early_stopping_monitor = create_early_stopping_monitor(args, [])

    if should_refresh_checkpoint_baseline(initialization):
        val_summary = validate(model, val_loader, criterion, device, use_amp)
        best_val_loss = val_summary["valLoss"]
        save_checkpoint(output_dir / "best_metric_model.pth", model, optimizer, scheduler, args, 0, val_summary, model_name)
        print_baseline_validation(val_summary)

    for epoch in range(start_epoch, args.max_epochs + 1):
        train_summary = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        scheduler.step()
        epoch_summary = {
            "epoch": epoch,
            "learningRate": scheduler.get_last_lr()[0],
            **train_summary
        }
        val_summary = {"validated": False}
        is_best = False

        if epoch == args.max_epochs or epoch % args.val_interval == 0:
            val_summary = validate(model, val_loader, criterion, device, use_amp)
            is_best = val_summary["valLoss"] < best_val_loss

            if is_best:
                best_val_loss = val_summary["valLoss"]

            epoch_summary.update(val_summary)

        early_stop_status = early_stopping_monitor.update(epoch, val_summary)
        epoch_summary["earlyStopping"] = early_stop_status

        if is_best:
            save_checkpoint(output_dir / "best_metric_model.pth", model, optimizer, scheduler, args, epoch, val_summary, model_name)

        print_training_progress(epoch, args.max_epochs, epoch_summary, val_summary, best_val_loss)

        if early_stop_status["shouldStop"]:
            print(f"Early stopping at epoch {epoch}: {early_stop_status['reason']}")
            break


def create_initialization_info():
    return {
        "mode": "scratch",
        "checkpointPath": None,
        "checkpointBestValLoss": None,
        "checkpointBestSsim": None,
        "checkpointSelectionMetric": None,
        "validation": None
    }


def load_model_weights(checkpoint_path, model, device, mode):
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["modelState"])
    checkpoint_val_loss = get_checkpoint_val_loss(checkpoint)
    initialization = {
        "mode": mode,
        "checkpointPath": str(checkpoint_path),
        "checkpointBestValLoss": checkpoint_val_loss if checkpoint_val_loss != float("inf") else None,
        "checkpointBestSsim": get_checkpoint_ssim(checkpoint),
        "checkpointSelectionMetric": checkpoint.get("selectionMetric"),
        "validation": create_checkpoint_validation_summary(checkpoint)
    }

    print(f"Initializing model weights from {checkpoint_path}")

    return checkpoint_val_loss, initialization


def should_refresh_checkpoint_baseline(initialization):
    if initialization["mode"] == "scratch":
        return False

    return initialization.get("checkpointSelectionMetric") != "valLoss"


def create_checkpoint_validation_summary(checkpoint):
    validation_summary = checkpoint.get("validation", {})

    if not validation_summary.get("validated"):
        return None

    return {
        "epoch": checkpoint.get("epoch"),
        **validation_summary
    }


def get_checkpoint_ssim(checkpoint):
    validation_summary = checkpoint.get("validation", {})

    if validation_summary.get("validated") and "ssim" in validation_summary:
        return validation_summary["ssim"]

    return -1.0


def get_checkpoint_val_loss(checkpoint):
    validation_summary = checkpoint.get("validation", {})

    if validation_summary.get("validated") and "valLoss" in validation_summary:
        return validation_summary["valLoss"]

    return float("inf")


def load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def create_data_loaders(args):
    train_dataset = Brain2DSliceDataset(args.data_dir, "train", augment=True, case_limit=args.train_case_limit)
    val_dataset = Brain2DSliceDataset(args.data_dir, "val", augment=False, case_limit=args.val_case_limit)
    use_persistent_workers = args.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=use_persistent_workers
    )

    return train_loader, val_loader, len(train_dataset), len(val_dataset)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    loss_values = []
    l1_values = []
    ssim_values = []
    gradient_values = []

    for batch in train_loader:
        input_image = batch["input"].to(device, non_blocking=True)
        target_image = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            prediction = model(input_image)
            loss, loss_summary = criterion(prediction, target_image)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_values.append(loss_summary["totalLoss"])
        l1_values.append(loss_summary["l1Loss"])
        ssim_values.append(loss_summary["ssimLoss"])
        gradient_values.append(loss_summary["gradientLoss"])

    return {
        "trainLoss": float(np.mean(loss_values)),
        "trainL1Loss": float(np.mean(l1_values)),
        "trainSsimLoss": float(np.mean(ssim_values)),
        "trainGradientLoss": float(np.mean(gradient_values))
    }


def validate(model, val_loader, criterion, device, use_amp):
    model.eval()
    loss_values = []
    metric_values = []

    with torch.no_grad():
        for batch in val_loader:
            input_image = batch["input"].to(device, non_blocking=True)
            target_image = batch["target"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                prediction = model(input_image)
                loss, loss_summary = criterion(prediction, target_image)

            loss_values.append(loss_summary["totalLoss"])
            metric_values.append(calculate_reconstruction_metrics(prediction.float(), target_image.float()))

    validation_summary = {
        "validated": True,
        "valLoss": float(np.mean(loss_values)),
        "mae": float(np.mean([metrics["mae"] for metrics in metric_values])),
        "rmse": float(np.mean([metrics["rmse"] for metrics in metric_values])),
        "psnr": float(np.mean([metrics["psnr"] for metrics in metric_values])),
        "ssim": float(np.mean([metrics["ssim"] for metrics in metric_values]))
    }

    for metric_name in ("foregroundMae", "foregroundRmse", "foregroundPsnr"):
        if metric_name in metric_values[0]:
            validation_summary[metric_name] = float(np.mean([metrics[metric_name] for metrics in metric_values]))

    return validation_summary


def save_training_config(output_dir, args, model_name, model_title, train_count, val_count, device, initialization):
    config_path = output_dir / "training_config.json"
    run_history = load_training_config_history(config_path)
    training_config = {
        "modelName": model_name,
        "modelTitle": model_title,
        "runStartedAt": datetime.now(timezone.utc).isoformat(),
        "dataDir": str(args.data_dir),
        "trainCount": train_count,
        "valCount": val_count,
        "device": str(device),
        "maxEpochs": args.max_epochs,
        "earlyStopping": {
            "valLossPatience": args.val_loss_patience,
            "metricPatience": args.metric_patience,
            "valLossMinDelta": args.val_loss_min_delta,
            "metricMinDelta": args.metric_min_delta,
            "warmupValidations": args.early_stop_warmup
        },
        "batchSize": args.batch_size,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "initialization": initialization,
        "loss": {
            "l1Weight": args.l1_weight,
            "ssimWeight": args.ssim_weight,
            "gradientWeight": args.gradient_weight,
            "foregroundWeight": getattr(args, "foreground_weight", 0.0),
            "backgroundWeight": getattr(args, "background_weight", 0.25),
            "foregroundThreshold": getattr(args, "foreground_threshold", 0.02)
        },
        "runHistory": run_history
    }
    write_json(config_path, training_config)


def load_training_config_history(config_path):
    if not config_path.exists():
        return []

    with config_path.open("r", encoding="utf-8") as json_file:
        previous_config = json.load(json_file)

    previous_runs = previous_config.get("runHistory", [])
    previous_config = {
        key: value
        for key, value in previous_config.items()
        if key != "runHistory"
    }

    return [*previous_runs, previous_config]


def save_checkpoint(checkpoint_path, model, optimizer, scheduler, args, epoch, val_summary, model_name):
    torch.save(
        {
            "modelName": model_name,
            "epoch": epoch,
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "schedulerState": scheduler.state_dict(),
            "trainingArgs": vars(args),
            "selectionMetric": "valLoss",
            "validation": val_summary
        },
        checkpoint_path
    )


def write_json(file_path, payload):
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def print_baseline_validation(val_summary):
    print(
        "Baseline checkpoint validation "
        f"valLoss={val_summary['valLoss']:.4f}"
        f" ssim={val_summary['ssim']:.4f}"
        f" psnr={val_summary['psnr']:.2f}"
        f" bestValLoss={val_summary['valLoss']:.4f}"
    )


def print_training_progress(epoch, total_epochs, epoch_summary, val_summary, best_val_loss):
    message = (
        f"Epoch [{epoch}/{total_epochs}] "
        f"trainLoss={epoch_summary['trainLoss']:.4f}"
    )

    if val_summary.get("validated"):
        message += (
            f" valLoss={val_summary['valLoss']:.4f}"
            f" ssim={val_summary['ssim']:.4f}"
            f" psnr={val_summary['psnr']:.2f}"
            f" bestValLoss={best_val_loss:.4f}"
        )

    print(message)
