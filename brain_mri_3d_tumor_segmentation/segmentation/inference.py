import shutil
from importlib import import_module
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import APIRouter, HTTPException, Query

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
PREDICTION_CACHE_VERSION = 1
UNCERTAINTY_REGION_ID = "uncertainty"
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


def get_prediction_response(case_id, model_id):
    selected_case, model_config = get_selected_case_and_model(case_id, model_id)
    prediction_manifest = ensure_prediction_cache(selected_case, model_config)
    prediction_manifest["cacheHit"] = prediction_manifest.get("cacheHit", False)

    return prediction_manifest


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

    normalized_volume = normalize_input_volume(np.moveaxis(original_volume, -1, 0))
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

    region_masks = create_region_masks(probabilities, model_config["threshold"])
    confidence_summary = create_confidence_summary(probabilities, region_masks, uncertainty_volume, model_config)

    return {
        "affine": nifti_image.affine,
        "header": nifti_image.header.copy(),
        "originalShape": list(original_volume.shape),
        "regions": region_masks,
        "uncertainty": uncertainty_volume,
        "confidenceSummary": confidence_summary,
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
    probability_std = stacked_probabilities.std(dim=0)[0]
    probabilities = mean_probabilities[0].detach().cpu().numpy()
    uncertainty_volume = probability_std.mean(dim=0).detach().cpu().numpy().astype(np.float32)

    return probabilities, uncertainty_volume


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
        "version": PREDICTION_CACHE_VERSION,
        "caseId": selected_case["caseId"],
        "fileName": selected_case["fileName"],
        "modelId": model_config["id"],
        "modelLabel": model_config["label"],
        "modelTitle": model_config["title"],
        "checkpoint": checkpoint_info,
        "inputShape": prediction_data["originalShape"],
        "outputShape": list(prediction_data["regions"]["combined"].shape),
        "threshold": model_config["threshold"],
        "regions": overlays,
        "regionCounts": region_counts,
        "confidenceSummary": prediction_data["confidenceSummary"],
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

    if manifest.get("version") != PREDICTION_CACHE_VERSION:
        return False

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


def get_prediction_cache_dir(model_id, case_id):
    return SEGMENTATION_CACHE_DIR / model_id / case_id / "prediction"


def get_prediction_nifti_url(model_id, case_id, region):
    return f"{STATIC_SEGMENTATION_CACHE_PATH}/{model_id}/{case_id}/prediction/{region}.nii.gz"


def clear_cache_dir(cache_dir):
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
