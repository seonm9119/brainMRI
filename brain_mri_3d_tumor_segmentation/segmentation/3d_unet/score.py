import argparse
import csv
import json
import math
from importlib import import_module
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    SpatialPadd
)
from scipy import ndimage
from torch.amp import autocast

from brain_mri_3d_tumor_segmentation.segmentation.data import (
    ConvertBratsLabelToRegionsd,
    REGION_NAMES,
    load_decathlon_cases,
    split_cases
)


SEGMENTATION_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SEGMENTATION_ROOT / "decathlon"
DEFAULT_CHECKPOINT_DIR = MODEL_DIR / "checkpoints" / "assignment_unet"
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "best_metric_model.pth"
DEFAULT_CONFIG_PATH = DEFAULT_CHECKPOINT_DIR / "training_config.json"
DEFAULT_OUTPUT_DIR = SEGMENTATION_ROOT / "segmentation_cache" / "assignment" / "score"
DEFAULT_MODEL_FACTORY = "brain_mri_3d_tumor_segmentation.segmentation.3d_unet.model:create_model"
DEFAULT_MODEL_TITLE = "MONAI 3D U-Net Baseline"
REGION_IDS = ("tc", "wt", "et")
REGION_NAME_BY_ID = dict(zip(REGION_IDS, REGION_NAMES))
TEST_LABEL_NOTE = "Decathlon imagesTs has no labels in this dataset, so test Dice/HD95 cannot be computed locally."


def parse_args():
    parser = argparse.ArgumentParser(description="Score the assignment baseline 3D U-Net on BRATS train and validation splits.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--split-config", help="Training config used only for train/validation split selection.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-factory", default=DEFAULT_MODEL_FACTORY)
    parser.add_argument("--model-title", default=DEFAULT_MODEL_TITLE)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--case-limit", type=int, help="Limit each evaluated split for a quick smoke test.")
    parser.add_argument("--train-case-limit", type=int, help="Override train case count for scoring only.")
    parser.add_argument("--val-case-limit", type=int, help="Override validation case count for scoring only.")
    parser.add_argument("--dry-run", action="store_true", help="Print split counts without model inference.")

    return parser.parse_args()


def main():
    command_args = parse_args()
    data_dir = Path(command_args.data_dir)
    checkpoint_path = Path(command_args.checkpoint)
    config_path = Path(command_args.config)
    output_dir = Path(command_args.output_dir)
    training_config = load_training_config(config_path)
    split_config_path = Path(command_args.split_config) if command_args.split_config else config_path
    split_config = load_training_config(split_config_path)
    cases_by_split = build_cases_by_split(data_dir, split_config, command_args)

    if command_args.dry_run:
        print(json.dumps(create_dry_run_summary(cases_by_split, data_dir, split_config), ensure_ascii=False, indent=2))
        return

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(command_args.device)
    model = load_checkpoint_model(checkpoint_path, device, command_args.model_factory)
    score_transform = create_score_transform(training_config)
    score_summary = create_score_summary(
        command_args.model_title,
        checkpoint_path,
        config_path,
        split_config_path,
        data_dir,
        command_args.threshold,
        device
    )
    all_case_metric_rows = []

    for split_name in command_args.splits:
        split_cases_for_scoring = cases_by_split[split_name]

        if split_name == "test":
            score_summary["splits"][split_name] = create_unlabeled_test_summary(split_cases_for_scoring)
            continue

        split_case_metric_rows = score_labeled_split(
            split_name,
            split_cases_for_scoring,
            model,
            score_transform,
            device,
            command_args.threshold,
            training_config,
            device.type == "cuda" and not command_args.no_amp
        )
        score_summary["splits"][split_name] = summarize_split_metrics(split_case_metric_rows)
        all_case_metric_rows.extend(split_case_metric_rows)

    write_json(output_dir / "score_summary.json", score_summary)
    write_case_metrics_csv(output_dir / "score_cases.csv", all_case_metric_rows)
    print(json.dumps(score_summary, ensure_ascii=False, indent=2))
    print(f"Wrote {output_dir / 'score_summary.json'}")
    print(f"Wrote {output_dir / 'score_cases.csv'}")


def load_training_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"training config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def build_cases_by_split(data_dir, training_config, command_args):
    training_cases = load_decathlon_cases(data_dir)
    train_cases, validation_cases = split_cases(training_cases, training_config["val_ratio"], training_config["seed"])
    train_case_limit = command_args.train_case_limit
    validation_case_limit = command_args.val_case_limit

    if train_case_limit is None:
        train_case_limit = training_config.get("train_case_limit")

    if validation_case_limit is None:
        validation_case_limit = training_config.get("val_case_limit")

    if train_case_limit:
        train_cases = train_cases[:train_case_limit]

    if validation_case_limit:
        validation_cases = validation_cases[:validation_case_limit]

    if command_args.case_limit:
        train_cases = train_cases[:command_args.case_limit]
        validation_cases = validation_cases[:command_args.case_limit]

    return {
        "train": train_cases,
        "val": validation_cases,
        "test": load_test_cases(data_dir, command_args.case_limit)
    }


def load_test_cases(data_dir, case_limit=None):
    dataset_path = data_dir / "dataset.json"

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        dataset_config = json.load(dataset_file)

    test_cases = []

    for test_image in dataset_config.get("test", []):
        image_path = data_dir / test_image.removeprefix("./")
        test_cases.append({
            "caseId": image_path.name.removesuffix(".nii.gz"),
            "imagePath": image_path,
            "labelPath": None
        })

    if case_limit:
        test_cases = test_cases[:case_limit]

    return test_cases


def create_dry_run_summary(cases_by_split, data_dir, training_config):
    split_summaries = {}

    for split_name, split_cases_for_scoring in cases_by_split.items():
        split_summaries[split_name] = {
            "caseCount": len(split_cases_for_scoring),
            "labelAvailable": split_name != "test",
            "sampleCaseIds": [selected_case["caseId"] for selected_case in split_cases_for_scoring[:5]]
        }

    return {
        "dataDir": str(data_dir),
        "seed": training_config["seed"],
        "valRatio": training_config["val_ratio"],
        "valCaseLimit": training_config.get("val_case_limit"),
        "splits": split_summaries,
        "testNote": TEST_LABEL_NOTE
    }


def create_score_summary(model_title, checkpoint_path, config_path, split_config_path, data_dir, threshold, device):
    return {
        "model": model_title,
        "checkpoint": str(checkpoint_path),
        "trainingConfig": str(config_path),
        "splitConfig": str(split_config_path),
        "dataDir": str(data_dir),
        "threshold": threshold,
        "device": str(device),
        "metricNotes": {
            "dice": "Mask overlap. Higher is better.",
            "hd95Mm": "95th percentile Hausdorff distance in millimeters. Lower is better.",
            "sensitivity": "Ground-truth tumor voxels recovered by prediction. Higher is better.",
            "absoluteVolumeErrorPct": "Absolute relative volume error. Lower is better."
        },
        "splits": {}
    }


def create_score_transform(training_config):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys="image", channel_dim=-1),
        EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
        ConvertBratsLabelToRegionsd(keys="label"),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=tuple(training_config["patch_size"])),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image", "label"])
    ])


def resolve_device(device_name):
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        return torch.device("cuda")

    if device_name == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_model(checkpoint_path, device, model_factory_path=DEFAULT_MODEL_FACTORY):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("modelState") or checkpoint
    model = load_model_factory(model_factory_path)().to(device)
    model.load_state_dict(model_state)
    model.eval()

    return model


def load_model_factory(model_factory_path):
    module_path, function_name = model_factory_path.split(":")
    model_module = import_module(module_path)

    return getattr(model_module, function_name)


def score_labeled_split(split_name, cases_for_scoring, model, score_transform, device, threshold, training_config, use_amp):
    split_case_metric_rows = []

    for case_index, selected_case in enumerate(cases_for_scoring, start=1):
        print(f"[{split_name}] {case_index}/{len(cases_for_scoring)} {selected_case['caseId']}", flush=True)
        transformed_case = score_transform({
            "caseId": selected_case["caseId"],
            "image": str(selected_case["imagePath"]),
            "label": str(selected_case["labelPath"])
        })
        image = transformed_case["image"].unsqueeze(0).to(device)
        label_regions = transformed_case["label"].detach().cpu().numpy().astype(bool)
        spacing = get_case_spacing(selected_case["imagePath"])

        with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
            logits = sliding_window_inference(
                inputs=image,
                roi_size=tuple(training_config["patch_size"]),
                sw_batch_size=training_config["sw_batch_size"],
                predictor=model,
                overlap=training_config["overlap"]
            )

        probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
        prediction_regions = create_prediction_regions(probabilities, threshold)
        case_metric_rows = score_case_regions(split_name, selected_case["caseId"], prediction_regions, label_regions, spacing)
        split_case_metric_rows.extend(case_metric_rows)

    return split_case_metric_rows


def get_case_spacing(image_path):
    zooms = nib.load(str(image_path)).header.get_zooms()

    return tuple(float(zoom) for zoom in zooms[:3])


def create_prediction_regions(probabilities, threshold):
    tumor_core = probabilities[0] >= threshold
    whole_tumor = probabilities[1] >= threshold
    enhancing_tumor = probabilities[2] >= threshold
    tumor_core = np.logical_or(tumor_core, enhancing_tumor)
    whole_tumor = np.logical_or(whole_tumor, tumor_core)

    return {
        "tc": tumor_core.astype(bool),
        "wt": whole_tumor.astype(bool),
        "et": enhancing_tumor.astype(bool)
    }


def score_case_regions(split_name, case_id, prediction_regions, label_regions, spacing):
    case_metric_rows = []
    label_regions_by_id = {
        "tc": label_regions[0],
        "wt": label_regions[1],
        "et": label_regions[2]
    }

    for region_id in REGION_IDS:
        prediction_mask = prediction_regions[region_id]
        label_mask = label_regions_by_id[region_id]
        case_metric_rows.append({
            "split": split_name,
            "caseId": case_id,
            "region": REGION_NAME_BY_ID[region_id],
            "dice": dice_score(prediction_mask, label_mask),
            "hd95Mm": hd95_score(prediction_mask, label_mask, spacing),
            "sensitivity": sensitivity_score(prediction_mask, label_mask),
            "volumeErrorPct": volume_error_pct(prediction_mask, label_mask),
            "absoluteVolumeErrorPct": absolute_volume_error_pct(prediction_mask, label_mask),
            "predictionVoxels": int(prediction_mask.sum()),
            "labelVoxels": int(label_mask.sum())
        })

    return case_metric_rows


def dice_score(prediction_mask, label_mask):
    prediction_voxel_count = int(prediction_mask.sum())
    label_voxel_count = int(label_mask.sum())

    if prediction_voxel_count == 0 and label_voxel_count == 0:
        return 1.0

    total_voxel_count = prediction_voxel_count + label_voxel_count

    if total_voxel_count == 0:
        return None

    overlap_voxel_count = int(np.logical_and(prediction_mask, label_mask).sum())

    return float((2 * overlap_voxel_count) / total_voxel_count)


def hd95_score(prediction_mask, label_mask, spacing):
    if not prediction_mask.any() and not label_mask.any():
        return 0.0

    if not prediction_mask.any() or not label_mask.any():
        return None

    prediction_surface = extract_surface(prediction_mask)
    label_surface = extract_surface(label_mask)

    if not prediction_surface.any() or not label_surface.any():
        return None

    prediction_to_label_distances = ndimage.distance_transform_edt(~label_surface, sampling=spacing)[prediction_surface]
    label_to_prediction_distances = ndimage.distance_transform_edt(~prediction_surface, sampling=spacing)[label_surface]
    surface_distances = np.concatenate([prediction_to_label_distances, label_to_prediction_distances])

    if surface_distances.size == 0:
        return None

    return float(np.percentile(surface_distances, 95))


def extract_surface(mask):
    if not mask.any():
        return mask.astype(bool)

    eroded_mask = ndimage.binary_erosion(mask, structure=np.ones((3, 3, 3)), border_value=0)

    return np.logical_xor(mask, eroded_mask)


def sensitivity_score(prediction_mask, label_mask):
    label_voxel_count = int(label_mask.sum())

    if label_voxel_count == 0:
        return 1.0 if not prediction_mask.any() else None

    true_positive_voxel_count = int(np.logical_and(prediction_mask, label_mask).sum())

    return float(true_positive_voxel_count / label_voxel_count)


def volume_error_pct(prediction_mask, label_mask):
    label_voxel_count = int(label_mask.sum())

    if label_voxel_count == 0:
        return 0.0 if not prediction_mask.any() else None

    prediction_voxel_count = int(prediction_mask.sum())

    return float(((prediction_voxel_count - label_voxel_count) / label_voxel_count) * 100)


def absolute_volume_error_pct(prediction_mask, label_mask):
    volume_error = volume_error_pct(prediction_mask, label_mask)

    if volume_error is None:
        return None

    return abs(volume_error)


def summarize_split_metrics(case_metric_rows):
    region_summaries = {}

    for region_name in REGION_NAMES:
        region_metric_rows = [metric_row for metric_row in case_metric_rows if metric_row["region"] == region_name]
        region_summaries[region_name] = {
            "caseCount": len(region_metric_rows),
            "diceMean": mean_metric(region_metric_rows, "dice"),
            "diceStd": std_metric(region_metric_rows, "dice"),
            "hd95MeanMm": mean_metric(region_metric_rows, "hd95Mm"),
            "hd95StdMm": std_metric(region_metric_rows, "hd95Mm"),
            "sensitivityMean": mean_metric(region_metric_rows, "sensitivity"),
            "absoluteVolumeErrorMeanPct": mean_metric(region_metric_rows, "absoluteVolumeErrorPct"),
            "validHd95CaseCount": count_valid_metrics(region_metric_rows, "hd95Mm")
        }

    return {
        "labelAvailable": True,
        "evaluatedCaseCount": len({metric_row["caseId"] for metric_row in case_metric_rows}),
        "regions": region_summaries
    }


def create_unlabeled_test_summary(test_cases):
    return {
        "labelAvailable": False,
        "caseCount": len(test_cases),
        "note": TEST_LABEL_NOTE,
        "sampleCaseIds": [selected_case["caseId"] for selected_case in test_cases[:5]]
    }


def mean_metric(metric_rows, metric_name):
    metric_values = valid_metric_values(metric_rows, metric_name)

    if not metric_values:
        return None

    return float(np.mean(metric_values))


def std_metric(metric_rows, metric_name):
    metric_values = valid_metric_values(metric_rows, metric_name)

    if not metric_values:
        return None

    return float(np.std(metric_values))


def count_valid_metrics(metric_rows, metric_name):
    return len(valid_metric_values(metric_rows, metric_name))


def valid_metric_values(metric_rows, metric_name):
    metric_values = []

    for metric_row in metric_rows:
        metric_value = metric_row.get(metric_name)

        if metric_value is None:
            continue

        if isinstance(metric_value, float) and (math.isnan(metric_value) or math.isinf(metric_value)):
            continue

        metric_values.append(metric_value)

    return metric_values


def write_json(file_path, payload):
    with file_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


def write_case_metrics_csv(file_path, case_metric_rows):
    fieldnames = [
        "split",
        "caseId",
        "region",
        "dice",
        "hd95Mm",
        "sensitivity",
        "volumeErrorPct",
        "absoluteVolumeErrorPct",
        "predictionVoxels",
        "labelVoxels"
    ]

    with file_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_metric_rows)


if __name__ == "__main__":
    main()
