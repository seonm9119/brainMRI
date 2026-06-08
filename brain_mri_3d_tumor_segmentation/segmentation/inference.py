import json
import os
import shutil
import urllib.error
import urllib.request
from importlib import import_module
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from PIL import Image, ImageDraw

from brain_mri_3d_tumor_segmentation.volume_3d.volume import (
    find_selected_case,
    get_case_path,
    get_modality_nifti_response,
    load_selected_cases,
    read_json,
    write_json
)


SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
SEGMENTATION_DIR = SEGMENTATION_ROOT / "segmentation"
ASSIGNMENT_CHECKPOINT_DIR = SEGMENTATION_DIR / "3d_unet" / "checkpoints"
ENHANCED_CHECKPOINT_DIR = SEGMENTATION_DIR / "enhanced_model" / "checkpoints"
SEGMENTATION_CACHE_DIR = SEGMENTATION_ROOT / "segmentation_cache"
STATIC_SEGMENTATION_CACHE_PATH = "/static/segmentation-cache"
UNCERTAINTY_REGION_ID = "uncertainty"
COMPARISON_SLICE_CACHE_VERSION = "full-brain-axis-v4"
GPT_INTERPRETATION_API_URL = os.environ.get(
    "BRAINMRI_GPT_INTERPRETATION_API_URL",
    "http://192.168.0.21:8003/brain-mri/segmentation/interpret"
).strip()
GPT_INTERPRETATION_TIMEOUT = float(os.environ.get("BRAINMRI_GPT_INTERPRETATION_TIMEOUT", "120"))
GPT_INTERPRETATION_MODEL_IDS = {"assignment", "enhanced"}
REGION_LABELS = {
    "wt": "WT",
    "tc": "TC",
    "et": "ET",
    "combined": "WT / TC / ET"
}
MODEL_CONFIGS = {
    "assignment": {
        "id": "assignment",
        "label": "과제 제출용",
        "title": "3D U-Net Baseline",
        "modelFactory": "brain_mri_3d_tumor_segmentation.segmentation.3d_unet.model:create_model",
        "checkpointCandidates": [
            ASSIGNMENT_CHECKPOINT_DIR / "assignment_unet" / "best_metric_model.pth",
            ASSIGNMENT_CHECKPOINT_DIR / "smoke_test_128_padded" / "best_metric_model.pth"
        ],
        "roiSize": (128, 128, 128),
        "swBatchSize": 2,
        "threshold": 0.5
    },
    "enhanced": {
        "id": "enhanced",
        "label": "개선 버전",
        "title": "Confidence-aware SwinUNETR",
        "modelFactory": "brain_mri_3d_tumor_segmentation.segmentation.enhanced_model.model:create_model",
        "checkpointCandidates": [
            ENHANCED_CHECKPOINT_DIR / "confidence_swin_unetr" / "best_metric_model.pth",
            ENHANCED_CHECKPOINT_DIR / "confidence_swin_unetr" / "latest_model.pth"
        ],
        "roiSize": (96, 96, 96),
        "swBatchSize": 1,
        "threshold": 0.5,
        "confidenceAware": True,
        "ttaFlipDims": [
            [],
            [2],
            [3],
            [4]
        ],
        "uncertaintyThreshold": 0.08
    }
}
MODEL_CACHE = {}
router = APIRouter(prefix="/api/brain-mri/segmentation", tags=["Segmentation Inference"])


@router.get("/cases/{case_id}/prediction")
def get_prediction(case_id: str, model: str = Query("assignment")):
    return get_prediction_response(case_id, model)


@router.get("/cases/{case_id}/prediction-difference")
def get_prediction_difference(
    case_id: str,
    model: str = Query("enhanced"),
    baseline: str = Query("assignment"),
    region: str = Query("tc")
):
    return get_prediction_difference_response(case_id, model, baseline, region)


@router.get("/cases/{case_id}/prediction-comparison-slice")
def get_prediction_comparison_slice(
    case_id: str,
    model: str = Query("enhanced"),
    baseline: str = Query("assignment"),
    region: str = Query("tc")
):
    return get_prediction_comparison_slice_response(case_id, model, baseline, region)


def get_prediction_response(case_id, model_id):
    selected_case, model_config = get_selected_case_and_model(case_id, model_id)
    prediction_manifest = ensure_prediction_cache(selected_case, model_config)
    prediction_manifest["cacheHit"] = prediction_manifest.get("cacheHit", False)

    return prediction_manifest


def get_prediction_difference_response(case_id, model_id, baseline_model_id, region_id):
    selected_case, model_config = get_selected_case_and_model(case_id, model_id)
    _, baseline_model_config = get_selected_case_and_model(case_id, baseline_model_id)
    normalized_region_id = normalize_difference_region_id(region_id)
    model_prediction_manifest = ensure_prediction_cache(selected_case, model_config)
    baseline_prediction_manifest = ensure_prediction_cache(selected_case, baseline_model_config)
    model_checkpoint = model_prediction_manifest["checkpoint"]
    baseline_checkpoint = baseline_prediction_manifest["checkpoint"]
    cache_dir = get_difference_cache_dir(model_config["id"], selected_case["caseId"])
    manifest_path = cache_dir / f"{baseline_model_config['id']}_vs_{model_config['id']}_{normalized_region_id}.json"

    if is_difference_manifest_ready(manifest_path, model_checkpoint, baseline_checkpoint):
        difference_manifest = read_json(manifest_path)
        difference_manifest["cacheHit"] = True
        return difference_manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    difference_manifest = save_prediction_difference_cache(
        cache_dir,
        selected_case,
        model_config,
        baseline_model_config,
        normalized_region_id,
        model_prediction_manifest,
        baseline_prediction_manifest
    )
    difference_manifest["cacheHit"] = False

    return difference_manifest


def get_prediction_comparison_slice_response(case_id, model_id, baseline_model_id, region_id):
    selected_case, model_config = get_selected_case_and_model(case_id, model_id)
    _, baseline_model_config = get_selected_case_and_model(case_id, baseline_model_id)
    normalized_region_id = normalize_difference_region_id(region_id)
    model_prediction_manifest = ensure_prediction_cache(selected_case, model_config)
    baseline_prediction_manifest = ensure_prediction_cache(selected_case, baseline_model_config)
    difference_manifest = get_prediction_difference_response(
        case_id,
        model_config["id"],
        baseline_model_config["id"],
        normalized_region_id
    )
    cache_dir = get_comparison_slice_cache_dir(model_config["id"], selected_case["caseId"])
    manifest_path = cache_dir / f"{baseline_model_config['id']}_vs_{model_config['id']}_{normalized_region_id}_slice.json"

    if is_comparison_slice_manifest_ready(
        manifest_path,
        model_prediction_manifest["checkpoint"],
        baseline_prediction_manifest["checkpoint"]
    ):
        comparison_manifest = read_json(manifest_path)
        comparison_manifest["cacheHit"] = True
        return comparison_manifest

    cache_dir.mkdir(parents=True, exist_ok=True)
    comparison_manifest = save_prediction_comparison_slice_cache(
        cache_dir,
        selected_case,
        model_config,
        baseline_model_config,
        normalized_region_id,
        model_prediction_manifest,
        baseline_prediction_manifest,
        difference_manifest
    )
    comparison_manifest["cacheHit"] = False

    return comparison_manifest


def get_selected_case_and_model(case_id, model_id):
    selected_cases = load_selected_cases()
    normalized_case_id = normalize_case_id(case_id)
    normalized_model_id = model_id.lower()

    if normalized_model_id not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"{model_id} model을 찾을 수 없습니다.")

    return find_selected_case(selected_cases, normalized_case_id), MODEL_CONFIGS[normalized_model_id]


def ensure_prediction_cache(selected_case, model_config):
    checkpoint_info = get_checkpoint_info(model_config)
    cache_dir = get_prediction_cache_dir(model_config["id"], selected_case["caseId"])
    manifest_path = cache_dir / "manifest.json"

    if is_prediction_manifest_ready(manifest_path, checkpoint_info):
        prediction_manifest = read_json(manifest_path)
        prediction_manifest["checkpoint"] = checkpoint_info
        prediction_manifest["cacheHit"] = True
        return prediction_manifest

    clear_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_data = run_model_inference(selected_case, model_config, checkpoint_info)
    prediction_manifest = save_prediction_cache(cache_dir, selected_case, model_config, checkpoint_info, prediction_data)
    prediction_manifest["cacheHit"] = False

    return prediction_manifest


def run_model_inference(selected_case, model_config, checkpoint_info):
    torch, sliding_window_inference = import_inference_dependencies()
    model = load_model(model_config, checkpoint_info, torch)
    case_path = get_case_path(selected_case)
    nifti_image = nib.load(str(case_path))
    original_volume = np.asarray(nifti_image.dataobj, dtype=np.float32)

    if original_volume.ndim != 4 or original_volume.shape[-1] != 4:
        raise HTTPException(status_code=422, detail="4-channel BRATS NIfTI volume만 inference할 수 있습니다.")

    spatial_shape = original_volume.shape[:3]
    channel_first_volume = np.moveaxis(original_volume, -1, 0)
    crop_slices = get_foreground_crop_slices(original_volume, model_config["roiSize"])
    cropped_volume = channel_first_volume[(slice(None),) + crop_slices]
    crop_metadata = create_crop_metadata(crop_slices, spatial_shape, cropped_volume.shape[1:])
    normalized_volume = normalize_input_volume(cropped_volume)
    input_tensor = torch.from_numpy(normalized_volume[None]).to(model["device"])

    if model_config.get("confidenceAware"):
        probabilities, uncertainty_volume = run_confidence_aware_inference(
            input_tensor,
            model,
            model_config,
            torch,
            sliding_window_inference
        )
    else:
        probability_tensor = run_probability_inference(input_tensor, model, model_config, torch, sliding_window_inference)
        probabilities = probability_tensor[0].detach().cpu().numpy()
        uncertainty_volume = None

    probabilities = restore_probability_volume(probabilities, spatial_shape, crop_slices)

    if uncertainty_volume is not None:
        uncertainty_volume = restore_scalar_volume(uncertainty_volume, spatial_shape, crop_slices)

    region_masks = create_region_masks(probabilities, model_config["threshold"])
    confidence_summary = create_confidence_summary(probabilities, region_masks, uncertainty_volume, model_config)
    quantitative_summary = create_quantitative_summary(original_volume, probabilities, region_masks, uncertainty_volume, nifti_image)
    llm_interpretation = create_llm_interpretation(selected_case, model_config, quantitative_summary)

    return {
        "affine": nifti_image.affine,
        "header": nifti_image.header.copy(),
        "originalShape": list(original_volume.shape),
        "inferenceCrop": crop_metadata,
        "regions": region_masks,
        "uncertainty": uncertainty_volume,
        "confidenceSummary": confidence_summary,
        "quantitativeSummary": quantitative_summary,
        "llmInterpretation": llm_interpretation,
        "probabilityRange": {
            "min": float(probabilities.min()),
            "max": float(probabilities.max())
        }
    }


def import_inference_dependencies():
    try:
        import torch
        from monai.inferers import sliding_window_inference
    except ImportError as import_error:
        raise HTTPException(
            status_code=500,
            detail="torch/monai inference dependency가 설치되어 있지 않습니다."
        ) from import_error

    return torch, sliding_window_inference


def run_probability_inference(input_tensor, model, model_config, torch, sliding_window_inference):
    use_amp = model["device"].type == "cuda"

    with torch.no_grad(), torch.amp.autocast(device_type=model["device"].type, enabled=use_amp):
        logits = sliding_window_inference(
            inputs=input_tensor,
            roi_size=model_config["roiSize"],
            sw_batch_size=model_config["swBatchSize"],
            predictor=model["network"],
            overlap=0.5
        )

    return torch.sigmoid(logits)


def run_confidence_aware_inference(input_tensor, model, model_config, torch, sliding_window_inference):
    probability_tensors = []

    for flip_dims in model_config["ttaFlipDims"]:
        tta_input_tensor = input_tensor

        if flip_dims:
            tta_input_tensor = torch.flip(input_tensor, dims=flip_dims)

        probability_tensor = run_probability_inference(tta_input_tensor, model, model_config, torch, sliding_window_inference)

        if flip_dims:
            probability_tensor = torch.flip(probability_tensor, dims=flip_dims)

        probability_tensors.append(probability_tensor)

    stacked_probabilities = torch.stack(probability_tensors, dim=0)
    mean_probabilities = stacked_probabilities.mean(dim=0)
    probability_std = stacked_probabilities.std(dim=0, unbiased=False)[0]
    probabilities = mean_probabilities[0].detach().cpu().numpy()
    uncertainty_volume = probability_std.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    return probabilities, uncertainty_volume


def get_foreground_crop_slices(original_volume, roi_size, margin=8):
    foreground_mask = create_brain_foreground_mask(original_volume)
    spatial_shape = original_volume.shape[:3]

    if not foreground_mask.any():
        return tuple(slice(0, axis_size) for axis_size in spatial_shape)

    foreground_indices = np.where(foreground_mask)
    crop_slices = []

    for axis, axis_size in enumerate(spatial_shape):
        axis_start = max(int(foreground_indices[axis].min()) - margin, 0)
        axis_end = min(int(foreground_indices[axis].max()) + margin + 1, axis_size)
        minimum_size = min(int(roi_size[axis]), axis_size)
        current_size = axis_end - axis_start

        if current_size < minimum_size:
            missing_size = minimum_size - current_size
            left_padding = missing_size // 2
            right_padding = missing_size - left_padding
            axis_start = max(axis_start - left_padding, 0)
            axis_end = min(axis_end + right_padding, axis_size)

            if axis_end - axis_start < minimum_size:
                axis_start = max(axis_end - minimum_size, 0)
                axis_end = min(axis_start + minimum_size, axis_size)

        crop_slices.append(slice(axis_start, axis_end))

    return tuple(crop_slices)


def create_crop_metadata(crop_slices, spatial_shape, cropped_shape):
    return {
        "start": [int(crop_slice.start or 0) for crop_slice in crop_slices],
        "end": [int(crop_slice.stop) for crop_slice in crop_slices],
        "originalSpatialShape": [int(axis_size) for axis_size in spatial_shape],
        "croppedSpatialShape": [int(axis_size) for axis_size in cropped_shape]
    }


def restore_probability_volume(cropped_probabilities, spatial_shape, crop_slices):
    restored_probabilities = np.zeros(
        (cropped_probabilities.shape[0],) + tuple(spatial_shape),
        dtype=cropped_probabilities.dtype
    )
    restored_probabilities[(slice(None),) + crop_slices] = cropped_probabilities

    return restored_probabilities


def restore_scalar_volume(cropped_volume, spatial_shape, crop_slices):
    restored_volume = np.zeros(tuple(spatial_shape), dtype=cropped_volume.dtype)
    restored_volume[crop_slices] = cropped_volume

    return restored_volume


def create_region_masks(probabilities, threshold):
    tumor_core = probabilities[0] >= threshold
    whole_tumor = probabilities[1] >= threshold
    enhancing_tumor = probabilities[2] >= threshold
    tumor_core = np.logical_or(tumor_core, enhancing_tumor)
    whole_tumor = np.logical_or(whole_tumor, tumor_core)
    combined_mask = np.zeros(whole_tumor.shape, dtype=np.uint8)
    combined_mask[whole_tumor] = 1
    combined_mask[tumor_core] = 2
    combined_mask[enhancing_tumor] = 3

    return {
        "wt": whole_tumor.astype(np.uint8),
        "tc": tumor_core.astype(np.uint8),
        "et": enhancing_tumor.astype(np.uint8),
        "combined": combined_mask
    }


def create_confidence_summary(probabilities, region_masks, uncertainty_volume, model_config):
    if uncertainty_volume is None:
        return None

    region_channels = {
        "wt": 1,
        "tc": 0,
        "et": 2
    }
    region_confidence = {}

    for region_id, channel_index in region_channels.items():
        region_mask = region_masks[region_id].astype(bool)
        region_confidence[region_id] = calculate_region_confidence(
            probabilities[channel_index],
            uncertainty_volume,
            region_mask
        )

    tumor_mask = region_masks["combined"] > 0
    uncertainty_threshold = model_config["uncertaintyThreshold"]
    uncertain_voxel_ratio = calculate_uncertain_voxel_ratio(uncertainty_volume, tumor_mask, uncertainty_threshold)
    review_recommended = uncertain_voxel_ratio >= 0.08 or region_confidence["et"]["confidence"] < 0.65

    return {
        "method": "flip TTA mean probability with prediction disagreement uncertainty",
        "ttaCount": len(model_config["ttaFlipDims"]),
        "uncertaintyThreshold": uncertainty_threshold,
        "uncertainVoxelRatio": uncertain_voxel_ratio,
        "reviewRecommended": review_recommended,
        "regions": region_confidence
    }


def calculate_region_confidence(region_probability, uncertainty_volume, region_mask):
    if not region_mask.any():
        return {
            "confidence": 0.0,
            "meanProbability": 0.0,
            "meanUncertainty": 0.0,
            "voxelCount": 0
        }

    mean_probability = float(region_probability[region_mask].mean())
    mean_uncertainty = float(uncertainty_volume[region_mask].mean())
    confidence = mean_probability * (1 - min(mean_uncertainty * 2, 1))

    return {
        "confidence": float(np.clip(confidence, 0, 1)),
        "meanProbability": mean_probability,
        "meanUncertainty": mean_uncertainty,
        "voxelCount": int(region_mask.sum())
    }


def calculate_uncertain_voxel_ratio(uncertainty_volume, tumor_mask, uncertainty_threshold):
    if not tumor_mask.any():
        return 0.0

    uncertain_voxels = np.logical_and(tumor_mask, uncertainty_volume >= uncertainty_threshold)

    return float(uncertain_voxels.sum() / tumor_mask.sum())


def create_quantitative_summary(original_volume, probabilities, region_masks, uncertainty_volume, nifti_image):
    brain_foreground_mask = create_brain_foreground_mask(original_volume)
    tumor_mask = region_masks["combined"] > 0
    brain_foreground_mask = np.logical_or(brain_foreground_mask, tumor_mask)
    voxel_spacing_mm = [float(zoom) for zoom in nifti_image.header.get_zooms()[:3]]
    voxel_volume_ml = calculate_voxel_volume_ml(voxel_spacing_mm)
    brain_voxel_count = int(brain_foreground_mask.sum())
    region_summaries = create_region_quantitative_summaries(
        region_masks,
        probabilities,
        brain_voxel_count,
        voxel_volume_ml
    )
    composition_summary = create_composition_summary(region_summaries, voxel_volume_ml)
    probability_uncertainty = create_probability_uncertainty_summary(probabilities, tumor_mask)
    tta_uncertainty = create_tta_uncertainty_summary(uncertainty_volume, tumor_mask)

    return {
        "method": "case-level quantitative summary from predicted WT/TC/ET masks",
        "voxelSpacingMm": voxel_spacing_mm,
        "voxelVolumeMl": voxel_volume_ml,
        "brainForeground": {
            "voxelCount": brain_voxel_count,
            "volumeMl": brain_voxel_count * voxel_volume_ml
        },
        "regions": region_summaries,
        "composition": composition_summary,
        "uncertainty": {
            "probability": probability_uncertainty,
            "tta": tta_uncertainty
        }
    }


def create_brain_foreground_mask(original_volume):
    clean_volume = np.nan_to_num(original_volume, nan=0, posinf=0, neginf=0)

    if clean_volume.ndim == 4:
        return np.any(clean_volume > 0, axis=-1)

    return clean_volume > 0


def calculate_voxel_volume_ml(voxel_spacing_mm):
    voxel_volume_mm3 = float(np.prod(voxel_spacing_mm))

    return voxel_volume_mm3 / 1000


def create_region_quantitative_summaries(region_masks, probabilities, brain_voxel_count, voxel_volume_ml):
    probability_channel_by_region = {
        "tc": 0,
        "wt": 1,
        "et": 2
    }
    region_summaries = {}

    for region_id in ("wt", "tc", "et"):
        region_mask = region_masks[region_id].astype(bool)
        region_voxel_count = int(region_mask.sum())
        probability_channel = probability_channel_by_region[region_id]
        mean_probability = 0.0

        if region_voxel_count:
            mean_probability = float(probabilities[probability_channel][region_mask].mean())

        region_summaries[region_id] = {
            "label": REGION_LABELS[region_id],
            "voxelCount": region_voxel_count,
            "volumeMl": region_voxel_count * voxel_volume_ml,
            "brainRatio": safe_ratio(region_voxel_count, brain_voxel_count),
            "meanProbability": mean_probability
        }

    return region_summaries


def create_composition_summary(region_summaries, voxel_volume_ml):
    whole_tumor_voxels = region_summaries["wt"]["voxelCount"]
    tumor_core_voxels = region_summaries["tc"]["voxelCount"]
    enhancing_tumor_voxels = region_summaries["et"]["voxelCount"]
    edema_related_voxels = max(whole_tumor_voxels - tumor_core_voxels, 0)
    non_enhancing_core_voxels = max(tumor_core_voxels - enhancing_tumor_voxels, 0)

    return {
        "tcWtRatio": safe_ratio(tumor_core_voxels, whole_tumor_voxels),
        "etTcRatio": safe_ratio(enhancing_tumor_voxels, tumor_core_voxels),
        "etWtRatio": safe_ratio(enhancing_tumor_voxels, whole_tumor_voxels),
        "edemaRelated": {
            "voxelCount": edema_related_voxels,
            "volumeMl": edema_related_voxels * voxel_volume_ml,
            "wtRatio": safe_ratio(edema_related_voxels, whole_tumor_voxels)
        },
        "nonEnhancingCore": {
            "voxelCount": non_enhancing_core_voxels,
            "volumeMl": non_enhancing_core_voxels * voxel_volume_ml,
            "tcRatio": safe_ratio(non_enhancing_core_voxels, tumor_core_voxels)
        }
    }


def create_probability_uncertainty_summary(probabilities, tumor_mask):
    region_uncertainty = 4 * probabilities * (1 - probabilities)
    uncertainty_volume = region_uncertainty.max(axis=0).astype(np.float32)

    return create_uncertainty_values(uncertainty_volume, tumor_mask, 0.5)


def create_tta_uncertainty_summary(uncertainty_volume, tumor_mask):
    if uncertainty_volume is None:
        return None

    return create_uncertainty_values(uncertainty_volume, tumor_mask, 0.08)


def create_uncertainty_values(uncertainty_volume, tumor_mask, high_uncertainty_threshold):
    if not tumor_mask.any():
        return {
            "mean": 0.0,
            "max": 0.0,
            "highRatio": 0.0,
            "threshold": high_uncertainty_threshold
        }

    tumor_uncertainty = uncertainty_volume[tumor_mask]

    return {
        "mean": float(tumor_uncertainty.mean()),
        "max": float(tumor_uncertainty.max()),
        "highRatio": float((tumor_uncertainty >= high_uncertainty_threshold).sum() / tumor_uncertainty.size),
        "threshold": high_uncertainty_threshold
    }


def safe_ratio(numerator, denominator):
    if not denominator:
        return 0.0

    return float(numerator / denominator)


def create_llm_interpretation(selected_case, model_config, quantitative_summary):
    if model_config["id"] not in GPT_INTERPRETATION_MODEL_IDS:
        return None

    if not GPT_INTERPRETATION_API_URL:
        return None

    request_payload = {
        "caseId": selected_case["caseId"],
        "modelTitle": model_config["title"],
        "quantitativeSummary": quantitative_summary,
        "scoreContext": load_score_context(model_config["id"]),
        "heatmapContext": load_heatmap_context(model_config["id"])
    }

    try:
        request_body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
        api_request = urllib.request.Request(
            GPT_INTERPRETATION_API_URL,
            data=request_body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(api_request, timeout=GPT_INTERPRETATION_TIMEOUT) as api_response:
            response_body = api_response.read().decode("utf-8")

        return json.loads(response_body)
    except urllib.error.HTTPError as http_error:
        return create_llm_interpretation_error("http", http_error)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as api_error:
        return create_llm_interpretation_error("network", api_error)


def create_llm_interpretation_error(error_type, api_error):
    error_message = str(api_error)

    if isinstance(api_error, urllib.error.HTTPError):
        try:
            error_message = api_error.read().decode("utf-8")
        except OSError:
            error_message = str(api_error)

    return {
        "success": False,
        "provider": "remote-gpt-api",
        "cached": False,
        "errorType": error_type,
        "message": error_message
    }


def load_score_context(model_id):
    score_path = SEGMENTATION_CACHE_DIR / model_id / "score" / "score_summary.json"

    if not score_path.exists():
        return {}

    score_summary = read_json(score_path)
    validation_regions = score_summary.get("splits", {}).get("val", {}).get("regions", {})
    score_context = {}

    for region_label in ("WT", "TC", "ET"):
        region_score = validation_regions.get(region_label, {})
        score_context[region_label] = {
            "diceMean": region_score.get("diceMean"),
            "hd95MeanMm": region_score.get("hd95MeanMm"),
            "sensitivityMean": region_score.get("sensitivityMean"),
            "absoluteVolumeErrorMeanPct": region_score.get("absoluteVolumeErrorMeanPct")
        }

    return {
        "split": "validation",
        "regions": score_context
    }


def load_heatmap_context(model_id):
    heatmap_path = SEGMENTATION_CACHE_DIR / model_id / "heatmap" / "manifest.json"

    if not heatmap_path.exists():
        return {}

    heatmap_manifest = read_json(heatmap_path)
    validation_summary = heatmap_manifest.get("validationSummary") or {}

    return {
        "caseId": heatmap_manifest.get("caseId"),
        "meanUncertainty": validation_summary.get("meanUncertainty", heatmap_manifest.get("meanUncertainty")),
        "maxUncertainty": validation_summary.get("maxUncertainty", heatmap_manifest.get("maxUncertainty")),
        "uncertainVoxelRatio": validation_summary.get("uncertainVoxelRatio", heatmap_manifest.get("uncertainVoxelRatio")),
        "caseCount": validation_summary.get("caseCount")
    }


def load_model(model_config, checkpoint_info, torch):
    cache_key = f"{model_config['id']}:{checkpoint_info['path']}:{checkpoint_info['mtime']}"

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_info["path"], map_location=device, weights_only=False)
    model_state = checkpoint.get("modelState") or checkpoint
    network = load_model_factory(model_config["modelFactory"])().to(device)
    network.load_state_dict(model_state)
    network.eval()
    MODEL_CACHE.clear()
    MODEL_CACHE[cache_key] = {
        "network": network,
        "device": device
    }

    return MODEL_CACHE[cache_key]


def load_model_factory(model_factory_path):
    module_path, function_name = model_factory_path.split(":")
    model_module = import_module(module_path)

    return getattr(model_module, function_name)


def normalize_input_volume(channel_first_volume):
    normalized_channels = []

    for channel_volume in channel_first_volume:
        clean_channel = np.nan_to_num(channel_volume, nan=0, posinf=0, neginf=0).astype(np.float32)
        foreground = clean_channel[clean_channel > 0]

        if foreground.size:
            mean = float(foreground.mean())
            std = float(foreground.std())
        else:
            mean = float(clean_channel.mean())
            std = float(clean_channel.std())

        if std < 1e-6:
            std = 1.0

        normalized_channel = (clean_channel - mean) / std
        normalized_channels.append(normalized_channel.astype(np.float32))

    return np.ascontiguousarray(np.stack(normalized_channels, axis=0))


def save_prediction_cache(cache_dir, selected_case, model_config, checkpoint_info, prediction_data):
    overlays = {}
    region_counts = {}

    for region_id, region_volume in prediction_data["regions"].items():
        region_path = cache_dir / f"{region_id}.nii.gz"
        save_region_nifti(region_path, region_volume, prediction_data["affine"], prediction_data["header"])
        region_counts[region_id] = int(region_volume.astype(bool).sum())
        overlays[region_id] = {
            "id": region_id,
            "label": REGION_LABELS[region_id],
            "niftiUrl": get_prediction_nifti_url(model_config["id"], selected_case["caseId"], region_id),
            "voxelCount": region_counts[region_id]
        }

    uncertainty_info = save_uncertainty_cache(cache_dir, selected_case, model_config, prediction_data)
    prediction_manifest = {
        "caseId": selected_case["caseId"],
        "fileName": selected_case["fileName"],
        "modelId": model_config["id"],
        "modelLabel": model_config["label"],
        "modelTitle": model_config["title"],
        "checkpoint": checkpoint_info,
        "inputShape": prediction_data["originalShape"],
        "outputShape": list(prediction_data["regions"]["combined"].shape),
        "inferenceCrop": prediction_data["inferenceCrop"],
        "threshold": model_config["threshold"],
        "regions": overlays,
        "regionCounts": region_counts,
        "confidenceSummary": prediction_data["confidenceSummary"],
        "quantitativeSummary": prediction_data["quantitativeSummary"],
        "llmInterpretation": prediction_data["llmInterpretation"],
        "uncertainty": uncertainty_info,
        "probabilityRange": prediction_data["probabilityRange"],
        "baseNifti": get_modality_nifti_response(selected_case["caseId"], "flair")
    }
    write_json(cache_dir / "manifest.json", prediction_manifest)

    return prediction_manifest


def save_uncertainty_cache(cache_dir, selected_case, model_config, prediction_data):
    uncertainty_volume = prediction_data.get("uncertainty")

    if uncertainty_volume is None:
        return None

    uncertainty_path = cache_dir / f"{UNCERTAINTY_REGION_ID}.nii.gz"
    save_float_nifti(
        uncertainty_path,
        uncertainty_volume,
        prediction_data["affine"],
        prediction_data["header"]
    )

    return {
        "id": UNCERTAINTY_REGION_ID,
        "label": "TTA uncertainty",
        "niftiUrl": get_prediction_nifti_url(model_config["id"], selected_case["caseId"], UNCERTAINTY_REGION_ID),
        "min": float(uncertainty_volume.min()),
        "max": float(uncertainty_volume.max()),
        "mean": float(uncertainty_volume.mean())
    }


def save_prediction_difference_cache(
    cache_dir,
    selected_case,
    model_config,
    baseline_model_config,
    region_id,
    model_prediction_manifest,
    baseline_prediction_manifest
):
    model_region_info = model_prediction_manifest["regions"][region_id]
    baseline_region_info = baseline_prediction_manifest["regions"][region_id]
    model_region_path = get_static_cache_file_path(model_region_info["niftiUrl"])
    baseline_region_path = get_static_cache_file_path(baseline_region_info["niftiUrl"])
    model_region_image = nib.load(str(model_region_path))
    baseline_region_image = nib.load(str(baseline_region_path))
    model_region_mask = np.asarray(model_region_image.dataobj) > 0
    baseline_region_mask = np.asarray(baseline_region_image.dataobj) > 0
    baseline_only_mask = np.logical_and(baseline_region_mask, ~model_region_mask)
    model_only_mask = np.logical_and(model_region_mask, ~baseline_region_mask)
    difference_volume = np.zeros(model_region_mask.shape, dtype=np.uint8)
    difference_volume[baseline_only_mask] = 1
    difference_volume[model_only_mask] = 2
    difference_filename = f"{baseline_model_config['id']}_vs_{model_config['id']}_{region_id}.nii.gz"
    difference_path = cache_dir / difference_filename
    save_region_nifti(difference_path, difference_volume, model_region_image.affine, model_region_image.header.copy())
    baseline_voxel_count = int(baseline_region_mask.sum())
    model_voxel_count = int(model_region_mask.sum())
    baseline_only_voxel_count = int(baseline_only_mask.sum())
    model_only_voxel_count = int(model_only_mask.sum())
    disagreement_voxel_count = baseline_only_voxel_count + model_only_voxel_count
    union_voxel_count = int(np.logical_or(baseline_region_mask, model_region_mask).sum())
    manifest = {
        "caseId": selected_case["caseId"],
        "fileName": selected_case["fileName"],
        "region": region_id,
        "regionLabel": REGION_LABELS[region_id],
        "modelId": model_config["id"],
        "modelTitle": model_config["title"],
        "baselineModelId": baseline_model_config["id"],
        "baselineModelTitle": baseline_model_config["title"],
        "modelCheckpoint": model_prediction_manifest["checkpoint"],
        "baselineCheckpoint": baseline_prediction_manifest["checkpoint"],
        "niftiUrl": get_difference_nifti_url(model_config["id"], selected_case["caseId"], difference_filename),
        "labels": {
            "1": "3D U-Net only",
            "2": "SwinUNETR only"
        },
        "counts": {
            "baselineVoxels": baseline_voxel_count,
            "modelVoxels": model_voxel_count,
            "baselineOnlyVoxels": baseline_only_voxel_count,
            "modelOnlyVoxels": model_only_voxel_count,
            "disagreementVoxels": disagreement_voxel_count,
            "unionVoxels": union_voxel_count
        },
        "ratios": {
            "baselineOnlyUnionRatio": safe_ratio(baseline_only_voxel_count, union_voxel_count),
            "modelOnlyUnionRatio": safe_ratio(model_only_voxel_count, union_voxel_count),
            "disagreementUnionRatio": safe_ratio(disagreement_voxel_count, union_voxel_count)
        }
    }
    manifest_path = cache_dir / f"{baseline_model_config['id']}_vs_{model_config['id']}_{region_id}.json"
    write_json(manifest_path, manifest)

    return manifest


def save_prediction_comparison_slice_cache(
    cache_dir,
    selected_case,
    model_config,
    baseline_model_config,
    region_id,
    model_prediction_manifest,
    baseline_prediction_manifest,
    difference_manifest
):
    case_path = get_case_path(selected_case)
    case_image = nib.load(str(case_path))
    original_volume = np.asarray(case_image.dataobj, dtype=np.float32)
    flair_volume = original_volume[:, :, :, 0]
    model_mask = load_prediction_mask(model_prediction_manifest["regions"][region_id]["niftiUrl"])
    baseline_mask = load_prediction_mask(baseline_prediction_manifest["regions"][region_id]["niftiUrl"])
    baseline_only_mask = np.logical_and(baseline_mask, ~model_mask)
    model_only_mask = np.logical_and(model_mask, ~baseline_mask)
    axis, slice_index = select_comparison_slice(
        flair_volume,
        baseline_mask,
        model_mask,
        baseline_only_mask,
        model_only_mask
    )
    image_slice = extract_display_slice(flair_volume, axis, slice_index)
    baseline_slice = extract_display_slice(baseline_mask.astype(np.uint8), axis, slice_index)
    model_slice = extract_display_slice(model_mask.astype(np.uint8), axis, slice_index)
    baseline_only_slice = extract_display_slice(baseline_only_mask.astype(np.uint8), axis, slice_index)
    model_only_slice = extract_display_slice(model_only_mask.astype(np.uint8), axis, slice_index)
    crop_box = get_comparison_crop_box(image_slice, baseline_slice, model_slice, baseline_only_slice, model_only_slice)
    image_slice = crop_slice(image_slice, crop_box)
    baseline_slice = crop_slice(baseline_slice, crop_box)
    model_slice = crop_slice(model_slice, crop_box)
    baseline_only_slice = crop_slice(baseline_only_slice, crop_box)
    model_only_slice = crop_slice(model_only_slice, crop_box)
    panel_size = 230
    baseline_color = get_region_rgb_color(region_id)
    model_color = get_region_rgb_color(region_id)
    comparison_image = create_model_comparison_slice_image([
        ("MRI", create_mri_slice_panel(image_slice, panel_size)),
        ("3D U-Net", create_mask_overlay_panel(image_slice, baseline_slice, baseline_color, panel_size)),
        ("SwinUNETR", create_mask_overlay_panel(image_slice, model_slice, model_color, panel_size)),
        ("Difference", create_difference_overlay_panel(image_slice, baseline_only_slice, model_only_slice, panel_size))
    ])
    image_filename = (
        f"{baseline_model_config['id']}_vs_{model_config['id']}_{region_id}_"
        f"{COMPARISON_SLICE_CACHE_VERSION}_slice.png"
    )
    image_path = cache_dir / image_filename
    comparison_image.save(image_path)
    comparison_manifest = {
        "cacheVersion": COMPARISON_SLICE_CACHE_VERSION,
        "caseId": selected_case["caseId"],
        "fileName": selected_case["fileName"],
        "region": region_id,
        "regionLabel": REGION_LABELS[region_id],
        "axis": axis,
        "sliceIndex": slice_index,
        "imageUrl": get_comparison_slice_image_url(model_config["id"], selected_case["caseId"], image_filename),
        "modelId": model_config["id"],
        "modelTitle": model_config["title"],
        "baselineModelId": baseline_model_config["id"],
        "baselineModelTitle": baseline_model_config["title"],
        "modelCheckpoint": model_prediction_manifest["checkpoint"],
        "baselineCheckpoint": baseline_prediction_manifest["checkpoint"],
        "difference": {
            "niftiUrl": difference_manifest["niftiUrl"],
            "counts": difference_manifest["counts"],
            "ratios": difference_manifest["ratios"],
            "labels": difference_manifest["labels"]
        },
        "interpretation": create_comparison_slice_interpretation(region_id, difference_manifest)
    }
    manifest_path = cache_dir / f"{baseline_model_config['id']}_vs_{model_config['id']}_{region_id}_slice.json"
    write_json(manifest_path, comparison_manifest)

    return comparison_manifest


def load_prediction_mask(nifti_url):
    return np.asarray(nib.load(str(get_static_cache_file_path(nifti_url))).dataobj) > 0


def select_comparison_slice(image_volume, baseline_mask, model_mask, baseline_only_mask, model_only_mask):
    disagreement_mask = np.logical_or(baseline_only_mask, model_only_mask)
    union_mask = np.logical_or(baseline_mask, model_mask)
    best_candidate = None

    for axis in range(3):
        foreground_counts = []
        slice_scores = []

        for slice_index in range(image_volume.shape[axis]):
            disagreement_slice = extract_raw_slice(disagreement_mask, axis, slice_index)
            union_slice = extract_raw_slice(union_mask, axis, slice_index)
            image_slice = extract_raw_slice(image_volume, axis, slice_index)
            image_foreground = np.abs(image_slice) > 1e-6
            foreground_count = float(image_foreground.sum())
            foreground_counts.append(foreground_count)
            slice_scores.append(
                float(disagreement_slice.sum()) * 1.0 +
                float(union_slice.sum()) * 0.18 +
                foreground_count * 0.012
            )

        if not slice_scores or max(slice_scores) <= 0:
            continue

        max_foreground_count = max(foreground_counts)
        minimum_foreground_count = max_foreground_count * 0.72
        eligible_slice_indices = [
            slice_index
            for slice_index, foreground_count in enumerate(foreground_counts)
            if foreground_count >= minimum_foreground_count
        ]

        if not eligible_slice_indices:
            eligible_slice_indices = list(range(len(slice_scores)))

        axis_best_slice_index = int(max(eligible_slice_indices, key=lambda slice_index: slice_scores[slice_index]))
        axis_best_score = float(slice_scores[axis_best_slice_index])
        axis_foreground_ratio = safe_ratio(foreground_counts[axis_best_slice_index], max_foreground_count)
        display_quality = calculate_display_foreground_quality(
            extract_display_slice(image_volume, axis, axis_best_slice_index)
        )
        axis_candidate_score = axis_best_score * (0.90 + axis_foreground_ratio * 0.10) * display_quality

        if best_candidate is None or axis_candidate_score > best_candidate["score"]:
            best_candidate = {
                "axis": axis,
                "sliceIndex": axis_best_slice_index,
                "score": axis_candidate_score
            }

    if best_candidate is None:
        return 2, int(image_volume.shape[2] // 2)

    return int(best_candidate["axis"]), int(best_candidate["sliceIndex"])


def extract_raw_slice(volume, axis, slice_index):
    if axis == 0:
        return volume[slice_index, :, :]

    if axis == 1:
        return volume[:, slice_index, :]

    return volume[:, :, slice_index]


def calculate_display_foreground_quality(image_slice):
    foreground_mask = np.abs(image_slice) > 1e-6

    if not foreground_mask.any():
        return 0.25

    row_indices, column_indices = np.where(foreground_mask)
    foreground_area = float(foreground_mask.sum())
    foreground_height = float(row_indices.max() - row_indices.min() + 1)
    foreground_width = float(column_indices.max() - column_indices.min() + 1)
    bounding_area = max(1.0, foreground_height * foreground_width)
    fill_score = min(1.0, (foreground_area / bounding_area) / 0.70)
    height_score = min(1.0, (foreground_height / float(image_slice.shape[0])) / 0.66)
    width_score = min(1.0, (foreground_width / float(image_slice.shape[1])) / 0.50)
    aspect_score = min(foreground_height / foreground_width, foreground_width / foreground_height)
    foreground_shape_score = height_score * width_score * fill_score * aspect_score

    return 0.25 + foreground_shape_score * 0.75


def extract_display_slice(volume, axis, slice_index):
    return np.rot90(extract_raw_slice(volume, axis, slice_index))


def get_comparison_crop_box(image_slice, *mask_slices):
    foreground_mask = np.abs(image_slice) > 1e-6

    for mask_slice in mask_slices:
        foreground_mask = np.logical_or(foreground_mask, mask_slice > 0)

    if not foreground_mask.any():
        return 0, image_slice.shape[0], 0, image_slice.shape[1]

    row_indices, column_indices = np.where(foreground_mask)
    row_padding = max(18, int(image_slice.shape[0] * 0.14))
    column_padding = max(18, int(image_slice.shape[1] * 0.14))
    row_start = max(int(row_indices.min()) - row_padding, 0)
    row_end = min(int(row_indices.max()) + row_padding + 1, image_slice.shape[0])
    column_start = max(int(column_indices.min()) - column_padding, 0)
    column_end = min(int(column_indices.max()) + column_padding + 1, image_slice.shape[1])

    return row_start, row_end, column_start, column_end


def crop_slice(slice_data, crop_box):
    row_start, row_end, column_start, column_end = crop_box

    return slice_data[row_start:row_end, column_start:column_end]


def create_mri_slice_panel(image_slice, panel_size):
    normalized_slice = normalize_mri_slice(image_slice)
    image = Image.fromarray((normalized_slice * 255).astype(np.uint8), mode="L").convert("RGB")

    return resize_panel(image, panel_size)


def create_mask_overlay_panel(image_slice, mask_slice, color, panel_size):
    panel = create_mri_slice_panel(image_slice, panel_size).convert("RGBA")
    resized_mask = resize_label_slice(mask_slice, panel_size)
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()

    for y in range(panel.height):
        for x in range(panel.width):
            if resized_mask[y, x] <= 0:
                continue

            overlay_pixels[x, y] = (*color, 132)

    panel = Image.alpha_composite(panel, overlay)
    draw_slice_edges(panel, resized_mask, (255, 255, 255, 220))

    return panel.convert("RGB")


def create_difference_overlay_panel(image_slice, baseline_only_slice, model_only_slice, panel_size):
    panel = create_mri_slice_panel(image_slice, panel_size).convert("RGBA")
    resized_baseline_only = resize_label_slice(baseline_only_slice, panel_size)
    resized_model_only = resize_label_slice(model_only_slice, panel_size)
    overlay = Image.new("RGBA", panel.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()

    for y in range(panel.height):
        for x in range(panel.width):
            if resized_baseline_only[y, x] > 0:
                overlay_pixels[x, y] = (81, 154, 255, 172)
            elif resized_model_only[y, x] > 0:
                overlay_pixels[x, y] = (255, 123, 67, 172)

    panel = Image.alpha_composite(panel, overlay)
    draw_slice_edges(panel, resized_baseline_only, (150, 195, 255, 230))
    draw_slice_edges(panel, resized_model_only, (255, 190, 140, 230))

    return panel.convert("RGB")


def create_model_comparison_slice_image(labelled_panels):
    panel_width = labelled_panels[0][1].width
    panel_height = labelled_panels[0][1].height
    title_height = 34
    gap = 10
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


def draw_slice_edges(panel, label_slice, color):
    label_mask = label_slice > 0

    if not label_mask.any():
        return

    from scipy import ndimage

    edge_mask = np.logical_xor(label_mask, ndimage.binary_erosion(label_mask, structure=np.ones((3, 3)), border_value=0))
    draw = ImageDraw.Draw(panel)

    for y, x in np.argwhere(edge_mask):
        draw.point((int(x), int(y)), fill=color)


def get_region_rgb_color(region_id):
    region_colors = {
        "wt": (37, 151, 150),
        "tc": (225, 83, 132),
        "et": (236, 184, 72),
        "combined": (185, 125, 226)
    }

    return region_colors.get(region_id, region_colors["combined"])


def create_comparison_slice_interpretation(region_id, difference_manifest):
    counts = difference_manifest["counts"]
    ratios = difference_manifest["ratios"]

    return {
        "summary": (
            f"{REGION_LABELS[region_id]} 기준으로 3D U-Net only {counts['baselineOnlyVoxels']:,} voxels, "
            f"SwinUNETR only {counts['modelOnlyVoxels']:,} voxels가 확인됩니다."
        ),
        "decision": (
            "테스트 샘플에는 정답 mask가 없으므로 더 넓은 영역을 잡은 모델을 곧바로 정답으로 볼 수 없습니다. "
            "validation 성능, 현재 case confidence, uncertainty 위치, MRI 신호 일관성을 함께 확인해야 합니다."
        ),
        "disagreementUnionRatio": ratios["disagreementUnionRatio"]
    }


def save_region_nifti(region_path, region_volume, affine, header):
    region_header = header.copy()
    region_header.set_data_shape(region_volume.shape)
    region_header.set_data_dtype(np.uint8)
    region_header.set_intent("label")
    region_header["cal_min"] = 0
    region_header["cal_max"] = int(region_volume.max())
    region_header["scl_slope"] = 1
    region_header["scl_inter"] = 0
    nib.save(nib.Nifti1Image(region_volume.astype(np.uint8), affine, region_header), str(region_path))


def save_float_nifti(file_path, volume, affine, header):
    float_header = header.copy()
    float_header.set_data_shape(volume.shape)
    float_header.set_data_dtype(np.float32)
    float_header["cal_min"] = float(volume.min())
    float_header["cal_max"] = float(volume.max())
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine, float_header), str(file_path))


def is_prediction_manifest_ready(manifest_path, checkpoint_info):
    if not manifest_path.exists():
        return False

    manifest = read_json(manifest_path)

    if manifest.get("checkpoint", {}).get("signature") != checkpoint_info["signature"]:
        return False

    for region_id in REGION_LABELS:
        region_info = manifest.get("regions", {}).get(region_id)

        if not region_info:
            return False

        region_path = SEGMENTATION_CACHE_DIR / region_info["niftiUrl"].removeprefix(f"{STATIC_SEGMENTATION_CACHE_PATH}/")

        if not region_path.exists():
            return False

    if manifest.get("modelId") == "enhanced":
        uncertainty_info = manifest.get("uncertainty")
        confidence_summary = manifest.get("confidenceSummary")

        if not uncertainty_info or not confidence_summary:
            return False

        uncertainty_path = SEGMENTATION_CACHE_DIR / uncertainty_info["niftiUrl"].removeprefix(f"{STATIC_SEGMENTATION_CACHE_PATH}/")

        if not uncertainty_path.exists():
            return False

    return True


def is_difference_manifest_ready(manifest_path, model_checkpoint, baseline_checkpoint):
    if not manifest_path.exists():
        return False

    manifest = read_json(manifest_path)

    if manifest.get("modelCheckpoint", {}).get("signature") != model_checkpoint["signature"]:
        return False

    if manifest.get("baselineCheckpoint", {}).get("signature") != baseline_checkpoint["signature"]:
        return False

    nifti_url = manifest.get("niftiUrl")

    if not nifti_url:
        return False

    return get_static_cache_file_path(nifti_url).exists()


def is_comparison_slice_manifest_ready(manifest_path, model_checkpoint, baseline_checkpoint):
    if not manifest_path.exists():
        return False

    manifest = read_json(manifest_path)

    if manifest.get("cacheVersion") != COMPARISON_SLICE_CACHE_VERSION:
        return False

    if manifest.get("modelCheckpoint", {}).get("signature") != model_checkpoint["signature"]:
        return False

    if manifest.get("baselineCheckpoint", {}).get("signature") != baseline_checkpoint["signature"]:
        return False

    image_url = manifest.get("imageUrl")

    if not image_url:
        return False

    return get_static_cache_file_path(image_url).exists()


def get_checkpoint_info(model_config):
    for checkpoint_path in model_config["checkpointCandidates"]:
        if checkpoint_path.exists():
            checkpoint_stat = checkpoint_path.stat()
            return {
                "path": str(checkpoint_path),
                "fileName": checkpoint_path.name,
                "size": checkpoint_stat.st_size,
                "mtime": checkpoint_stat.st_mtime,
                "signature": f"{checkpoint_path.name}:{checkpoint_stat.st_size}:{int(checkpoint_stat.st_mtime)}"
            }

    raise HTTPException(status_code=404, detail="사용 가능한 segmentation checkpoint를 찾을 수 없습니다.")


def normalize_case_id(case_id):
    return case_id.removesuffix(".nii.gz")


def normalize_difference_region_id(region_id):
    normalized_region_id = region_id.lower()

    if normalized_region_id == "all":
        return "combined"

    if normalized_region_id not in REGION_LABELS:
        raise HTTPException(status_code=422, detail=f"{region_id} region 차이 mask를 만들 수 없습니다.")

    return normalized_region_id


def get_prediction_cache_dir(model_id, case_id):
    return SEGMENTATION_CACHE_DIR / model_id / case_id / "prediction"


def get_difference_cache_dir(model_id, case_id):
    return SEGMENTATION_CACHE_DIR / model_id / case_id / "difference"


def get_comparison_slice_cache_dir(model_id, case_id):
    return SEGMENTATION_CACHE_DIR / model_id / case_id / "comparison"


def get_prediction_nifti_url(model_id, case_id, region):
    return f"{STATIC_SEGMENTATION_CACHE_PATH}/{model_id}/{case_id}/prediction/{region}.nii.gz"


def get_difference_nifti_url(model_id, case_id, filename):
    return f"{STATIC_SEGMENTATION_CACHE_PATH}/{model_id}/{case_id}/difference/{filename}"


def get_comparison_slice_image_url(model_id, case_id, filename):
    return f"{STATIC_SEGMENTATION_CACHE_PATH}/{model_id}/{case_id}/comparison/{filename}"


def get_static_cache_file_path(nifti_url):
    return SEGMENTATION_CACHE_DIR / nifti_url.removeprefix(f"{STATIC_SEGMENTATION_CACHE_PATH}/")


def clear_cache_dir(cache_dir):
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
