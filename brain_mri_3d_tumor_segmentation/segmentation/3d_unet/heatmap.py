import argparse
import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from PIL import Image, ImageDraw
from torch.amp import autocast

from brain_mri_3d_tumor_segmentation.segmentation.data import load_decathlon_cases, split_cases
from .score import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_CONFIG_PATH,
    REGION_IDS,
    REGION_NAME_BY_ID,
    create_prediction_regions,
    create_score_transform,
    get_case_spacing,
    load_checkpoint_model,
    load_training_config,
    resolve_device,
    score_case_regions,
    summarize_split_metrics
)


SEGMENTATION_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = SEGMENTATION_ROOT / "decathlon"
DEFAULT_OUTPUT_DIR = SEGMENTATION_ROOT / "segmentation_cache" / "assignment" / "heatmap"
STATIC_HEATMAP_PATH = "/static/segmentation-cache/assignment/heatmap"
REGION_COLORS = {
    "wt": (37, 151, 150),
    "tc": (225, 83, 132),
    "et": (236, 184, 72)
}
COMBINED_LABEL_BY_REGION = {
    "wt": 1,
    "tc": 2,
    "et": 3
}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a validation uncertainty heatmap for the assignment 3D U-Net.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--case-id", help="BRATS case id. Defaults to the first validation case from the trained split.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--axis", choices=["0", "1", "2"], default="2")
    parser.add_argument("--panel-size", type=int, default=320)
    parser.add_argument("--no-amp", action="store_true")

    return parser.parse_args()


def main():
    command_args = parse_args()
    data_dir = Path(command_args.data_dir)
    checkpoint_path = Path(command_args.checkpoint)
    config_path = Path(command_args.config)
    output_dir = Path(command_args.output_dir)
    training_config = load_training_config(config_path)
    train_cases, validation_cases = load_train_validation_cases(data_dir, training_config)
    selected_case = select_heatmap_case(train_cases, validation_cases, command_args.case_id)
    device = resolve_device(command_args.device)
    model = load_checkpoint_model(checkpoint_path, device)
    score_transform = create_score_transform(training_config)
    heatmap_data = run_heatmap_inference(
        selected_case,
        model,
        score_transform,
        device,
        training_config,
        command_args.threshold,
        device.type == "cuda" and not command_args.no_amp
    )
    validation_summary = create_validation_uncertainty_summary(
        validation_cases,
        selected_case,
        heatmap_data,
        model,
        score_transform,
        device,
        training_config,
        command_args.threshold,
        device.type == "cuda" and not command_args.no_amp
    )
    heatmap_manifest = save_heatmap_outputs(
        output_dir,
        selected_case,
        heatmap_data,
        int(command_args.axis),
        command_args.panel_size,
        validation_summary
    )

    print(json.dumps(heatmap_manifest, ensure_ascii=False, indent=2))


def load_train_validation_cases(data_dir, training_config):
    training_cases = load_decathlon_cases(data_dir)
    train_cases, validation_cases = split_cases(training_cases, training_config["val_ratio"], training_config["seed"])
    validation_case_limit = training_config.get("val_case_limit")

    if validation_case_limit:
        validation_cases = validation_cases[:validation_case_limit]

    return train_cases, validation_cases


def select_heatmap_case(train_cases, validation_cases, case_id):
    if not case_id:
        return validation_cases[0]

    normalized_case_id = case_id.removesuffix(".nii.gz")

    for selected_case in train_cases + validation_cases:
        if selected_case["caseId"] == normalized_case_id:
            return selected_case

    raise ValueError(f"{case_id} case를 찾을 수 없습니다.")


def run_heatmap_inference(selected_case, model, score_transform, device, training_config, threshold, use_amp):
    transformed_case = score_transform({
        "caseId": selected_case["caseId"],
        "image": str(selected_case["imagePath"]),
        "label": str(selected_case["labelPath"])
    })
    image = transformed_case["image"].unsqueeze(0).to(device)
    image_channels = transformed_case["image"].detach().cpu().numpy()
    label_regions = transformed_case["label"].detach().cpu().numpy().astype(bool)

    with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
        logits = sliding_window_inference(
            inputs=image,
            roi_size=tuple(training_config["patch_size"]),
            sw_batch_size=training_config["sw_batch_size"],
            predictor=model,
            overlap=training_config["overlap"]
        )

    probabilities = torch.sigmoid(logits)[0].detach().cpu().numpy()
    uncertainty_by_region = 4 * probabilities * (1 - probabilities)
    uncertainty_volume = uncertainty_by_region.max(axis=0).astype(np.float32)
    prediction_regions = create_prediction_regions(probabilities, threshold)
    prediction_combined = create_combined_region_volume(prediction_regions)
    label_combined = create_label_combined_region_volume(label_regions)
    spacing = get_case_spacing(selected_case["imagePath"])
    case_metric_rows = score_case_regions(
        "val",
        selected_case["caseId"],
        prediction_regions,
        label_regions,
        spacing
    )

    return {
        "image": image_channels[0],
        "predictionCombined": prediction_combined,
        "labelCombined": label_combined,
        "uncertainty": uncertainty_volume,
        "metrics": summarize_split_metrics(case_metric_rows)
    }


def create_combined_region_volume(region_masks):
    combined_volume = np.zeros(region_masks["wt"].shape, dtype=np.uint8)
    combined_volume[region_masks["wt"]] = COMBINED_LABEL_BY_REGION["wt"]
    combined_volume[region_masks["tc"]] = COMBINED_LABEL_BY_REGION["tc"]
    combined_volume[region_masks["et"]] = COMBINED_LABEL_BY_REGION["et"]

    return combined_volume


def create_label_combined_region_volume(label_regions):
    label_by_region = {
        "tc": label_regions[0],
        "wt": label_regions[1],
        "et": label_regions[2]
    }

    return create_combined_region_volume(label_by_region)


def create_validation_uncertainty_summary(
    validation_cases,
    selected_case,
    selected_heatmap_data,
    model,
    score_transform,
    device,
    training_config,
    threshold,
    use_amp
):
    case_summaries = []

    for case_index, validation_case in enumerate(validation_cases, start=1):
        print(f"[heatmap-val] {case_index}/{len(validation_cases)} {validation_case['caseId']}", flush=True)

        if validation_case["caseId"] == selected_case["caseId"]:
            heatmap_data = selected_heatmap_data
        else:
            heatmap_data = run_heatmap_inference(
                validation_case,
                model,
                score_transform,
                device,
                training_config,
                threshold,
                use_amp
            )

        case_summary = calculate_case_uncertainty_summary(heatmap_data)
        case_summary["caseId"] = validation_case["caseId"]
        case_summaries.append(case_summary)

    mean_values = np.asarray([case_summary["meanUncertainty"] for case_summary in case_summaries], dtype=np.float32)
    max_values = np.asarray([case_summary["maxUncertainty"] for case_summary in case_summaries], dtype=np.float32)
    high_ratio_values = np.asarray([case_summary["uncertainVoxelRatio"] for case_summary in case_summaries], dtype=np.float32)

    return {
        "scope": "validation split",
        "caseCount": len(case_summaries),
        "threshold": 0.5,
        "meanUncertainty": float(mean_values.mean()) if mean_values.size else 0.0,
        "meanUncertaintyStd": float(mean_values.std()) if mean_values.size else 0.0,
        "maxUncertaintyMean": float(max_values.mean()) if max_values.size else 0.0,
        "maxUncertainty": float(max_values.max()) if max_values.size else 0.0,
        "uncertainVoxelRatio": float(high_ratio_values.mean()) if high_ratio_values.size else 0.0,
        "uncertainVoxelRatioStd": float(high_ratio_values.std()) if high_ratio_values.size else 0.0,
        "caseSummaries": case_summaries
    }


def calculate_case_uncertainty_summary(heatmap_data):
    uncertainty_volume = heatmap_data["uncertainty"]
    tumor_mask = np.logical_or(heatmap_data["labelCombined"] > 0, heatmap_data["predictionCombined"] > 0)

    if not tumor_mask.any():
        return {
            "meanUncertainty": 0.0,
            "maxUncertainty": 0.0,
            "uncertainVoxelRatio": 0.0,
            "tumorVoxelCount": 0
        }

    tumor_uncertainty = uncertainty_volume[tumor_mask]

    return {
        "meanUncertainty": float(tumor_uncertainty.mean()),
        "maxUncertainty": float(tumor_uncertainty.max()),
        "uncertainVoxelRatio": float((tumor_uncertainty >= 0.5).sum() / tumor_uncertainty.size),
        "tumorVoxelCount": int(tumor_mask.sum())
    }


def save_heatmap_outputs(output_dir, selected_case, heatmap_data, axis, panel_size, validation_summary):
    case_output_dir = output_dir / selected_case["caseId"]
    case_output_dir.mkdir(parents=True, exist_ok=True)
    slice_index = select_representative_slice(
        heatmap_data["labelCombined"],
        heatmap_data["predictionCombined"],
        heatmap_data["uncertainty"],
        axis
    )
    image_slice = extract_display_slice(heatmap_data["image"], axis, slice_index)
    prediction_slice = extract_display_slice(heatmap_data["predictionCombined"], axis, slice_index)
    label_slice = extract_display_slice(heatmap_data["labelCombined"], axis, slice_index)
    uncertainty_slice = extract_display_slice(heatmap_data["uncertainty"], axis, slice_index)
    crop_box = get_slice_crop_box(image_slice, prediction_slice, label_slice)
    image_slice = crop_slice(image_slice, crop_box)
    prediction_slice = crop_slice(prediction_slice, crop_box)
    label_slice = crop_slice(label_slice, crop_box)
    uncertainty_slice = crop_slice(uncertainty_slice, crop_box)
    base_panel = create_base_mri_panel(image_slice, panel_size)
    prediction_panel = create_prediction_overlay_panel(image_slice, prediction_slice, label_slice, panel_size)
    uncertainty_panel = create_uncertainty_panel(image_slice, uncertainty_slice, panel_size)
    triptych = create_triptych_image([
        ("MRI", base_panel),
        ("Prediction / GT", prediction_panel),
        ("Uncertainty", uncertainty_panel)
    ])

    base_panel.save(case_output_dir / "mri.png")
    prediction_panel.save(case_output_dir / "prediction_overlay.png")
    uncertainty_panel.save(case_output_dir / "uncertainty_heatmap.png")
    triptych.save(case_output_dir / "heatmap_triptych.png")

    heatmap_manifest = create_heatmap_manifest(selected_case, heatmap_data, axis, slice_index, validation_summary)
    write_json(case_output_dir / "manifest.json", heatmap_manifest)
    write_json(output_dir / "manifest.json", heatmap_manifest)

    return heatmap_manifest


def select_representative_slice(label_combined, prediction_combined, uncertainty_volume, axis):
    tumor_volume = np.logical_or(label_combined > 0, prediction_combined > 0)
    uncertainty_weight = normalize_volume(uncertainty_volume)
    slice_scores = []

    for slice_index in range(tumor_volume.shape[axis]):
        tumor_slice = extract_raw_slice(tumor_volume, axis, slice_index)
        uncertainty_slice = extract_raw_slice(uncertainty_weight, axis, slice_index)
        slice_scores.append(float(tumor_slice.sum()) + float(uncertainty_slice.sum()) * 0.12)

    if max(slice_scores) <= 0:
        return int(tumor_volume.shape[axis] // 2)

    return int(np.argmax(slice_scores))


def extract_raw_slice(volume, axis, slice_index):
    if axis == 0:
        return volume[slice_index, :, :]

    if axis == 1:
        return volume[:, slice_index, :]

    return volume[:, :, slice_index]


def extract_display_slice(volume, axis, slice_index):
    raw_slice = extract_raw_slice(volume, axis, slice_index)

    return np.rot90(raw_slice)


def get_slice_crop_box(image_slice, prediction_slice, label_slice):
    foreground_mask = np.abs(image_slice) > 1e-6
    foreground_mask = np.logical_or(foreground_mask, prediction_slice > 0)
    foreground_mask = np.logical_or(foreground_mask, label_slice > 0)

    if not foreground_mask.any():
        return 0, image_slice.shape[0], 0, image_slice.shape[1]

    row_indices, column_indices = np.where(foreground_mask)
    row_padding = max(10, int(image_slice.shape[0] * 0.08))
    column_padding = max(10, int(image_slice.shape[1] * 0.08))
    row_start = max(int(row_indices.min()) - row_padding, 0)
    row_end = min(int(row_indices.max()) + row_padding + 1, image_slice.shape[0])
    column_start = max(int(column_indices.min()) - column_padding, 0)
    column_end = min(int(column_indices.max()) + column_padding + 1, image_slice.shape[1])

    return row_start, row_end, column_start, column_end


def crop_slice(slice_data, crop_box):
    row_start, row_end, column_start, column_end = crop_box

    return slice_data[row_start:row_end, column_start:column_end]


def create_base_mri_panel(image_slice, panel_size):
    grayscale = normalize_mri_slice(image_slice)
    image = Image.fromarray((grayscale * 255).astype(np.uint8), mode="L").convert("RGB")

    return resize_panel(image, panel_size)


def create_prediction_overlay_panel(image_slice, prediction_slice, label_slice, panel_size):
    panel = create_base_mri_panel(image_slice, panel_size).convert("RGBA")
    resized_prediction = resize_label_slice(prediction_slice, panel_size)
    resized_label = resize_label_slice(label_slice, panel_size)
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()

    for y in range(panel.height):
        for x in range(panel.width):
            label_value = resized_prediction[y, x]

            if label_value <= 0:
                continue

            color = get_combined_label_color(label_value)
            overlay_pixels[x, y] = (*color, 128)

    panel = Image.alpha_composite(panel, overlay)
    draw_region_edges(panel, resized_label, (255, 255, 255, 210))

    return panel.convert("RGB")


def create_uncertainty_panel(image_slice, uncertainty_slice, panel_size):
    base_panel = create_base_mri_panel(image_slice, panel_size).convert("RGBA")
    resized_uncertainty = resize_float_slice(uncertainty_slice, panel_size)
    normalized_uncertainty = normalize_volume(resized_uncertainty)
    heatmap = Image.new("RGBA", base_panel.size, (0, 0, 0, 0))
    heatmap_pixels = heatmap.load()

    for y in range(base_panel.height):
        for x in range(base_panel.width):
            uncertainty_value = float(normalized_uncertainty[y, x])

            if uncertainty_value <= 0.02:
                continue

            color = get_heatmap_color(uncertainty_value)
            alpha = int(np.clip(uncertainty_value * 220, 0, 220))
            heatmap_pixels[x, y] = (*color, alpha)

    return Image.alpha_composite(base_panel, heatmap).convert("RGB")


def create_triptych_image(labelled_panels):
    panel_width = labelled_panels[0][1].width
    panel_height = labelled_panels[0][1].height
    title_height = 34
    gap = 12
    canvas_width = panel_width * len(labelled_panels) + gap * (len(labelled_panels) - 1)
    canvas_height = panel_height + title_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), (248, 251, 253))
    draw = ImageDraw.Draw(canvas)

    for panel_index, (label, panel) in enumerate(labelled_panels):
        panel_x = panel_index * (panel_width + gap)
        draw.rounded_rectangle(
            [panel_x, 0, panel_x + panel_width, canvas_height - 1],
            radius=8,
            fill=(255, 255, 255),
            outline=(214, 228, 238)
        )
        draw.text((panel_x + 12, 10), label, fill=(8, 37, 61))
        canvas.paste(panel, (panel_x, title_height))

    return canvas


def draw_region_edges(panel, label_slice, color):
    label_mask = label_slice > 0

    if not label_mask.any():
        return

    edge_mask = np.logical_xor(label_mask, nd_binary_erosion(label_mask))
    draw = ImageDraw.Draw(panel)

    for y, x in np.argwhere(edge_mask):
        draw.point((int(x), int(y)), fill=color)


def nd_binary_erosion(mask):
    from scipy import ndimage

    return ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def resize_panel(image, panel_size):
    image = resize_image_to_fit_panel(image, panel_size, Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (panel_size, panel_size), (3, 12, 18))
    paste_x = (panel_size - image.width) // 2
    paste_y = (panel_size - image.height) // 2
    panel.paste(image, (paste_x, paste_y))

    return panel


def resize_label_slice(label_slice, panel_size):
    label_image = Image.fromarray(label_slice.astype(np.uint8), mode="L")
    label_image = resize_slice_to_panel(label_image, panel_size, Image.Resampling.NEAREST)

    return np.asarray(label_image, dtype=np.uint8)


def resize_float_slice(float_slice, panel_size):
    clean_slice = np.nan_to_num(float_slice, nan=0, posinf=0, neginf=0).astype(np.float32)
    scaled_slice = np.clip(clean_slice, 0, 1)
    float_image = Image.fromarray((scaled_slice * 255).astype(np.uint8), mode="L")
    resized_image = resize_slice_to_panel(float_image, panel_size, Image.Resampling.BILINEAR)

    return np.asarray(resized_image, dtype=np.float32) / 255


def resize_slice_to_panel(slice_image, panel_size, resample_mode):
    image = resize_image_to_fit_panel(slice_image, panel_size, resample_mode)
    panel = Image.new("L", (panel_size, panel_size), 0)
    paste_x = (panel_size - image.width) // 2
    paste_y = (panel_size - image.height) // 2
    panel.paste(image, (paste_x, paste_y))

    return panel


def resize_image_to_fit_panel(image, panel_size, resample_mode):
    width_scale = panel_size / image.width
    height_scale = panel_size / image.height
    scale = min(width_scale, height_scale)
    resized_width = max(1, int(image.width * scale))
    resized_height = max(1, int(image.height * scale))

    return image.resize((resized_width, resized_height), resample_mode)


def normalize_mri_slice(image_slice):
    clean_slice = np.nan_to_num(image_slice, nan=0, posinf=0, neginf=0).astype(np.float32)
    foreground_mask = np.abs(clean_slice) > 1e-6
    foreground = clean_slice[foreground_mask]

    if foreground.size:
        lower_bound = np.percentile(foreground, 1)
        upper_bound = np.percentile(foreground, 99)
    else:
        lower_bound = float(clean_slice.min())
        upper_bound = float(clean_slice.max())

    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 1

    normalized_slice = np.clip((clean_slice - lower_bound) / (upper_bound - lower_bound), 0, 1)
    normalized_slice[~foreground_mask] = 0

    return normalized_slice


def normalize_volume(volume):
    clean_volume = np.nan_to_num(volume, nan=0, posinf=0, neginf=0).astype(np.float32)
    upper_bound = np.percentile(clean_volume, 99.5)

    if upper_bound <= 0:
        return np.zeros_like(clean_volume, dtype=np.float32)

    return np.clip(clean_volume / upper_bound, 0, 1)


def get_combined_label_color(label_value):
    if label_value == COMBINED_LABEL_BY_REGION["wt"]:
        return REGION_COLORS["wt"]

    if label_value == COMBINED_LABEL_BY_REGION["tc"]:
        return REGION_COLORS["tc"]

    return REGION_COLORS["et"]


def get_heatmap_color(uncertainty_value):
    color_stops = [
        (25, 29, 74),
        (52, 76, 161),
        (20, 158, 177),
        (248, 205, 75),
        (247, 83, 93)
    ]
    scaled_position = np.clip(uncertainty_value, 0, 1) * (len(color_stops) - 1)
    lower_index = int(np.floor(scaled_position))
    upper_index = min(lower_index + 1, len(color_stops) - 1)
    fraction = scaled_position - lower_index
    lower_color = np.array(color_stops[lower_index], dtype=np.float32)
    upper_color = np.array(color_stops[upper_index], dtype=np.float32)
    color = lower_color + (upper_color - lower_color) * fraction

    return tuple(int(channel) for channel in color)


def create_heatmap_manifest(selected_case, heatmap_data, axis, slice_index, validation_summary):
    representative_uncertainty = calculate_case_uncertainty_summary(heatmap_data)
    case_static_path = f"{STATIC_HEATMAP_PATH}/{selected_case['caseId']}"

    return {
        "caseId": selected_case["caseId"],
        "fileName": selected_case["imagePath"].name,
        "split": "validation",
        "axis": axis,
        "sliceIndex": slice_index,
        "method": "max region uncertainty from 4 * p * (1 - p)",
        "imageUrl": f"{case_static_path}/mri.png",
        "predictionOverlayUrl": f"{case_static_path}/prediction_overlay.png",
        "uncertaintyHeatmapUrl": f"{case_static_path}/uncertainty_heatmap.png",
        "triptychUrl": f"{case_static_path}/heatmap_triptych.png",
        "meanUncertainty": representative_uncertainty["meanUncertainty"],
        "maxUncertainty": representative_uncertainty["maxUncertainty"],
        "uncertainVoxelRatio": representative_uncertainty["uncertainVoxelRatio"],
        "representativeUncertainty": representative_uncertainty,
        "validationSummary": validation_summary,
        "metrics": heatmap_data["metrics"],
        "regions": [
            {
                "id": region_id,
                "label": REGION_NAME_BY_ID[region_id],
                "color": REGION_COLORS[region_id]
            }
            for region_id in REGION_IDS
        ]
    }


def write_json(file_path, payload):
    with file_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
