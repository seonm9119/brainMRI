import json
import random
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from brain_mri_2d_slice_reconstruction.reconstruction.data import Brain2DSliceDataset
from brain_mri_2d_slice_reconstruction.reconstruction.early_stopping import create_early_stopping_monitor
from brain_mri_2d_slice_reconstruction.reconstruction.metrics import ReconstructionLoss, calculate_reconstruction_metrics


def run_gan_training(args, output_dir, create_generator, create_discriminator, model_name, model_title):
    set_seed(args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_count, val_count = create_data_loaders(args)
    generator = create_generator().to(device)
    discriminator = create_discriminator().to(device)
    reconstruction_criterion = ReconstructionLoss(
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        gradient_weight=args.gradient_weight,
        foreground_weight=getattr(args, "foreground_weight", 0.0),
        background_weight=getattr(args, "background_weight", 0.25),
        foreground_threshold=getattr(args, "foreground_threshold", 0.02)
    ).to(device)
    generator_optimizer = torch.optim.AdamW(
        generator.parameters(),
        lr=args.generator_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay
    )
    discriminator_optimizer = torch.optim.AdamW(
        discriminator.parameters(),
        lr=args.discriminator_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.weight_decay
    )
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, T_max=args.max_epochs)
    discriminator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(discriminator_optimizer, T_max=args.max_epochs)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = GradScaler(device.type, enabled=use_amp)
    best_val_loss = float("inf")
    initialization = create_initialization_info()
    start_epoch = 1

    if getattr(args, "initialize_from", None):
        best_val_loss, initialization = load_gan_weights(args.initialize_from, generator, discriminator, device, "explicit_checkpoint_finetune")
    elif (output_dir / "best_metric_model.pth").exists():
        best_val_loss, initialization = load_gan_weights(output_dir / "best_metric_model.pth", generator, discriminator, device, "best_checkpoint_finetune")

    save_training_config(output_dir, args, model_name, model_title, train_count, val_count, device, initialization)
    early_stopping_monitor = create_early_stopping_monitor(args, [])

    for epoch in range(start_epoch, args.max_epochs + 1):
        train_summary = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            reconstruction_criterion,
            generator_optimizer,
            discriminator_optimizer,
            scaler,
            device,
            use_amp,
            args.adversarial_weight
        )
        generator_scheduler.step()
        discriminator_scheduler.step()
        epoch_summary = {
            "epoch": epoch,
            "generatorLearningRate": generator_scheduler.get_last_lr()[0],
            "discriminatorLearningRate": discriminator_scheduler.get_last_lr()[0],
            **train_summary
        }
        val_summary = {"validated": False}
        is_best = False

        if epoch == args.max_epochs or epoch % args.val_interval == 0:
            val_summary = validate(generator, val_loader, reconstruction_criterion, device, use_amp)
            is_best = val_summary["valLoss"] < best_val_loss

            if is_best:
                best_val_loss = val_summary["valLoss"]

            epoch_summary.update(val_summary)

        early_stop_status = early_stopping_monitor.update(epoch, val_summary)
        epoch_summary["earlyStopping"] = early_stop_status

        if is_best:
            save_checkpoint(
                output_dir / "best_metric_model.pth",
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
                generator_scheduler,
                discriminator_scheduler,
                args,
                epoch,
                val_summary,
                model_name
            )

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
        "validation": None
    }


def load_gan_weights(checkpoint_path, generator, discriminator, device, mode):
    checkpoint = load_checkpoint(checkpoint_path, device)
    generator.load_state_dict(checkpoint["modelState"])

    if "discriminatorState" in checkpoint:
        discriminator.load_state_dict(checkpoint["discriminatorState"])

    checkpoint_val_loss = get_checkpoint_val_loss(checkpoint)
    initialization = {
        "mode": mode,
        "checkpointPath": str(checkpoint_path),
        "checkpointBestValLoss": checkpoint_val_loss if checkpoint_val_loss != float("inf") else None,
        "checkpointBestSsim": get_checkpoint_ssim(checkpoint),
        "validation": create_checkpoint_validation_summary(checkpoint)
    }

    print(f"Initializing GAN weights from {checkpoint_path}")

    return checkpoint_val_loss, initialization


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


def train_one_epoch(
    generator,
    discriminator,
    train_loader,
    reconstruction_criterion,
    generator_optimizer,
    discriminator_optimizer,
    scaler,
    device,
    use_amp,
    adversarial_weight
):
    generator.train()
    discriminator.train()
    generator_losses = []
    discriminator_losses = []
    adversarial_losses = []
    reconstruction_losses = []

    for batch in train_loader:
        input_image = batch["input"].to(device, non_blocking=True)
        target_image = batch["target"].to(device, non_blocking=True)

        set_requires_grad(discriminator, True)
        discriminator_optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            generated_image = generator(input_image)
            real_score = discriminator(input_image, target_image)
            fake_score = discriminator(input_image, generated_image.detach())
            discriminator_loss = (
                calculate_lsgan_loss(real_score, True) +
                calculate_lsgan_loss(fake_score, False)
            ) * 0.5

        scaler.scale(discriminator_loss).backward()
        scaler.step(discriminator_optimizer)

        set_requires_grad(discriminator, False)
        generator_optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            fake_score_for_generator = discriminator(input_image, generated_image)
            adversarial_loss = calculate_lsgan_loss(fake_score_for_generator, True)
            reconstruction_loss, reconstruction_summary = reconstruction_criterion(generated_image, target_image)
            generator_loss = reconstruction_loss + adversarial_weight * adversarial_loss

        scaler.scale(generator_loss).backward()
        scaler.step(generator_optimizer)
        scaler.update()
        set_requires_grad(discriminator, True)

        generator_losses.append(float(generator_loss.detach().cpu()))
        discriminator_losses.append(float(discriminator_loss.detach().cpu()))
        adversarial_losses.append(float(adversarial_loss.detach().cpu()))
        reconstruction_losses.append(reconstruction_summary["totalLoss"])

    return {
        "trainGeneratorLoss": float(np.mean(generator_losses)),
        "trainDiscriminatorLoss": float(np.mean(discriminator_losses)),
        "trainAdversarialLoss": float(np.mean(adversarial_losses)),
        "trainReconstructionLoss": float(np.mean(reconstruction_losses))
    }


def validate(generator, val_loader, reconstruction_criterion, device, use_amp):
    generator.eval()
    loss_values = []
    metric_values = []

    with torch.no_grad():
        for batch in val_loader:
            input_image = batch["input"].to(device, non_blocking=True)
            target_image = batch["target"].to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=use_amp):
                prediction = generator(input_image)
                loss, loss_summary = reconstruction_criterion(prediction, target_image)

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


def calculate_lsgan_loss(score, target_is_real):
    if isinstance(score, (list, tuple)):
        return sum(calculate_lsgan_loss(score_item, target_is_real) for score_item in score) / len(score)

    target_value = 1.0 if target_is_real else 0.0
    target_tensor = torch.full_like(score, target_value)

    return F.mse_loss(score, target_tensor)


def set_requires_grad(model, requires_grad):
    for parameter in model.parameters():
        parameter.requires_grad = requires_grad


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
        "generatorLearningRate": args.generator_learning_rate,
        "discriminatorLearningRate": args.discriminator_learning_rate,
        "weightDecay": args.weight_decay,
        "initialization": initialization,
        "loss": {
            "adversarialWeight": args.adversarial_weight,
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


def save_checkpoint(
    checkpoint_path,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    generator_scheduler,
    discriminator_scheduler,
    args,
    epoch,
    val_summary,
    model_name
):
    torch.save(
        {
            "modelName": model_name,
            "epoch": epoch,
            "modelState": generator.state_dict(),
            "discriminatorState": discriminator.state_dict(),
            "generatorOptimizerState": generator_optimizer.state_dict(),
            "discriminatorOptimizerState": discriminator_optimizer.state_dict(),
            "generatorSchedulerState": generator_scheduler.state_dict(),
            "discriminatorSchedulerState": discriminator_scheduler.state_dict(),
            "trainingArgs": vars(args),
            "selectionMetric": "valLoss",
            "validation": val_summary
        },
        checkpoint_path
    )


def write_json(file_path, payload):
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def print_training_progress(epoch, total_epochs, epoch_summary, val_summary, best_val_loss):
    message = (
        f"Epoch [{epoch}/{total_epochs}] "
        f"gLoss={epoch_summary['trainGeneratorLoss']:.4f} "
        f"dLoss={epoch_summary['trainDiscriminatorLoss']:.4f}"
    )

    if val_summary.get("validated"):
        message += (
            f" valLoss={val_summary['valLoss']:.4f}"
            f" ssim={val_summary['ssim']:.4f}"
            f" psnr={val_summary['psnr']:.2f}"
            f" bestValLoss={best_val_loss:.4f}"
        )

    print(message)
