import importlib
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import Image

from brain_mri_2d_slice_reconstruction.reconstruction.data import (
    INPUT_CHANNELS,
    TARGET_CHANNEL,
    get_reconstruction_case,
    load_case_arrays,
    load_reconstruction_cases,
    validate_split_name
)
from brain_mri_2d_slice_reconstruction.reconstruction.metrics import calculate_reconstruction_metrics


RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTION_DIR = Path(__file__).resolve().parent
MAMBA_CHECKPOINT_DIR = RECONSTRUCTION_DIR / "mamba_conv_unet" / "checkpoints"
PLAIN_UNET_CHECKPOINT_DIR = RECONSTRUCTION_DIR / "plain_unet" / "checkpoints"
PLAIN_GAN_CHECKPOINT_DIR = RECONSTRUCTION_DIR / "plain_gan" / "checkpoints"
RESVIT_GAN_CHECKPOINT_DIR = RECONSTRUCTION_DIR / "resvit_gan" / "checkpoints"
RECONSTRUCTION_CACHE_DIR = RECONSTRUCTION_ROOT / "reconstruction_cache"
STATIC_RECONSTRUCTION_CACHE_PATH = "/static/reconstruction-cache"
DEFAULT_DATA_DIR = RECONSTRUCTION_ROOT / "brain_2d"
MODEL_CACHE = {}

MODEL_CONFIGS = {
    "plain_unet": {
        "id": "plain_unet",
        "label": "Plain U-Net",
        "title": "Plain U-Net Baseline for Brain MRI Synthesis",
        "source": "Convolutional U-Net baseline",
        "modelFactory": "brain_mri_2d_slice_reconstruction.reconstruction.plain_unet.model:create_model",
        "checkpointCandidates": [
            PLAIN_UNET_CHECKPOINT_DIR / "plain_unet" / "best_metric_model.pth"
        ]
    },
    "mamba_conv_unet": {
        "id": "mamba_conv_unet",
        "label": "Mamba-Conv U-Net",
        "title": "Mamba-Conv U-Net for Brain MRI Synthesis",
        "source": "Mamba-inspired U-shaped encoder-decoder",
        "modelFactory": "brain_mri_2d_slice_reconstruction.reconstruction.mamba_conv_unet.model:create_model",
        "checkpointCandidates": [
            MAMBA_CHECKPOINT_DIR / "mamba_conv_unet" / "best_metric_model.pth"
        ]
    },
    "plain_gan": {
        "id": "plain_gan",
        "label": "Plain cGAN",
        "title": "Plain cGAN Pix2Pix Baseline for Brain MRI Synthesis",
        "source": "Pix2Pix-style conditional GAN baseline",
        "modelFactory": "brain_mri_2d_slice_reconstruction.reconstruction.plain_gan.model:create_model",
        "checkpointCandidates": [
            PLAIN_GAN_CHECKPOINT_DIR / "plain_gan" / "best_metric_model.pth"
        ]
    },
    "resvit_gan": {
        "id": "resvit_gan",
        "label": "ResViT-GAN",
        "title": "ResViT-GAN for Brain MRI Synthesis",
        "source": "Transformer bottleneck conditional GAN",
        "modelFactory": "brain_mri_2d_slice_reconstruction.reconstruction.resvit_gan.model:create_model",
        "checkpointCandidates": [
            RESVIT_GAN_CHECKPOINT_DIR / "resvit_gan" / "best_metric_model.pth"
        ]
    }
}

router = APIRouter(prefix="/api/brain-mri/reconstruction", tags=["2D MRI Reconstruction"])


@router.get("/models")
def get_models():
    return {
        "models": [
            {
                "id": model_config["id"],
                "label": model_config["label"],
                "title": model_config["title"],
                "source": model_config["source"],
                "checkpoint": get_optional_checkpoint_info(model_config)
            }
            for model_config in MODEL_CONFIGS.values()
        ]
    }


@router.get("/dataset/summary")
def get_dataset_summary():
    split_summaries = {}

    for split_name in ("train", "val", "test"):
        split_cases = load_reconstruction_cases(DEFAULT_DATA_DIR, split_name)
        split_summaries[split_name] = {
            "caseCount": len(split_cases),
            "fileCount": len(split_cases) * 2
        }

    return {
        "dataDir": str(DEFAULT_DATA_DIR),
        "inputChannels": list(INPUT_CHANNELS),
        "targetChannel": TARGET_CHANNEL,
        "splits": split_summaries
    }


@router.get("/cases")
def get_cases(split="test", limit=24):
    try:
        selected_cases = load_reconstruction_cases(DEFAULT_DATA_DIR, split)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    selected_limit = max(1, min(int(limit), 200))

    return {
        "split": split,
        "totalCount": len(selected_cases),
        "cases": [
            {
                "caseId": selected_case["caseId"],
                "split": selected_case["split"]
            }
            for selected_case in selected_cases[:selected_limit]
        ]
    }


@router.get("/cases/{case_id}/sample")
def get_case_sample(case_id, split="test"):
    selected_case = get_selected_case(split, case_id)
    return ensure_sample_cache(selected_case)


@router.get("/cases/{case_id}/prediction")
def get_prediction(case_id, model="mamba_conv_unet", split="test"):
    selected_case = get_selected_case(split, case_id)
    model_config = get_model_config(model)
    checkpoint_info = get_checkpoint_info(model_config)

    return ensure_prediction_cache(selected_case, model_config, checkpoint_info)


def get_selected_case(split_name, case_id):
    try:
        validate_split_name(split_name)
        return get_reconstruction_case(DEFAULT_DATA_DIR, split_name, case_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


def get_model_config(model_id):
    normalized_model_id = model_id.strip().lower()

    if normalized_model_id not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 reconstruction model입니다: {model_id}")

    return MODEL_CONFIGS[normalized_model_id]


def ensure_sample_cache(selected_case):
    cache_dir = get_sample_cache_dir(selected_case["split"], selected_case["caseId"])
    manifest_path = cache_dir / "manifest.json"

    if manifest_path.exists():
        return read_json(manifest_path)

    input_array, target_array = load_case_arrays(selected_case)
    clear_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_images = save_input_channel_images(cache_dir, selected_case, input_array)
    save_grayscale_image(cache_dir / "target.png", target_array[0])
    manifest = {
        "caseId": selected_case["caseId"],
        "split": selected_case["split"],
        "inputShape": list(input_array.shape),
        "targetShape": list(target_array.shape),
        "inputChannels": input_images,
        "target": {
            "label": TARGET_CHANNEL,
            "imageUrl": get_sample_image_url(selected_case["split"], selected_case["caseId"], "target.png")
        }
    }
    write_json(manifest_path, manifest)

    return manifest


def ensure_prediction_cache(selected_case, model_config, checkpoint_info):
    cache_dir = get_prediction_cache_dir(model_config["id"], selected_case["split"], selected_case["caseId"])
    manifest_path = cache_dir / "manifest.json"

    if is_prediction_manifest_ready(manifest_path, checkpoint_info):
        prediction_manifest = read_json(manifest_path)
        prediction_manifest["cacheHit"] = True
        add_prediction_image_versions(prediction_manifest, checkpoint_info)
        return prediction_manifest

    prediction_data = run_model_inference(selected_case, model_config, checkpoint_info)
    clear_cache_dir(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    input_images = save_input_channel_images(cache_dir, selected_case, prediction_data["input"])
    save_grayscale_image(cache_dir / "target.png", prediction_data["target"][0])
    save_grayscale_image(cache_dir / "prediction.png", prediction_data["prediction"][0])
    save_error_image(cache_dir / "error.png", prediction_data["absoluteError"][0])
    manifest = {
        "caseId": selected_case["caseId"],
        "split": selected_case["split"],
        "modelId": model_config["id"],
        "modelTitle": model_config["title"],
        "checkpoint": checkpoint_info,
        "cacheHit": False,
        "inputShape": list(prediction_data["input"].shape),
        "targetShape": list(prediction_data["target"].shape),
        "predictionShape": list(prediction_data["prediction"].shape),
        "inputChannels": input_images,
        "target": {
            "label": TARGET_CHANNEL,
            "imageUrl": get_prediction_image_url(model_config["id"], selected_case["split"], selected_case["caseId"], "target.png")
        },
        "prediction": {
            "label": "Synthetic T1Gd",
            "imageUrl": get_prediction_image_url(model_config["id"], selected_case["split"], selected_case["caseId"], "prediction.png")
        },
        "errorMap": {
            "label": "Absolute error",
            "imageUrl": get_prediction_image_url(model_config["id"], selected_case["split"], selected_case["caseId"], "error.png")
        },
        "metrics": prediction_data["metrics"]
    }
    add_prediction_image_versions(manifest, checkpoint_info)
    write_json(manifest_path, manifest)

    return manifest


def run_model_inference(selected_case, model_config, checkpoint_info):
    torch = import_inference_torch()
    input_array, target_array = load_case_arrays(selected_case)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_config, checkpoint_info, torch, device)
    input_tensor = torch.from_numpy(input_array).unsqueeze(0).to(device)
    target_tensor = torch.from_numpy(target_array).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction_tensor = model(input_tensor).float().clamp(0, 1)

    metrics = calculate_reconstruction_metrics(prediction_tensor.cpu(), target_tensor.cpu())
    prediction_array = prediction_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    absolute_error = np.abs(prediction_array - target_array).astype(np.float32)

    return {
        "input": input_array,
        "target": target_array,
        "prediction": prediction_array,
        "absoluteError": absolute_error,
        "metrics": metrics
    }


def import_inference_torch():
    try:
        import torch
    except ImportError as error:
        raise HTTPException(status_code=500, detail="PyTorch를 import하지 못했습니다.") from error

    return torch


def load_model(model_config, checkpoint_info, torch, device):
    cache_key = (model_config["id"], checkpoint_info["path"], checkpoint_info["modifiedTime"])

    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    model_factory = load_model_factory(model_config["modelFactory"])
    model = model_factory().to(device)
    checkpoint = torch.load(checkpoint_info["path"], map_location=device)
    model_state = checkpoint.get("modelState", checkpoint)
    model.load_state_dict(model_state)
    model.eval()
    MODEL_CACHE.clear()
    MODEL_CACHE[cache_key] = model

    return model


def load_model_factory(model_factory_path):
    module_path, function_name = model_factory_path.split(":")
    module = importlib.import_module(module_path)

    return getattr(module, function_name)


def save_input_channel_images(cache_dir, selected_case, input_array):
    input_images = []

    for channel_index, channel_name in enumerate(INPUT_CHANNELS):
        file_name = f"input_{channel_index}.png"
        save_grayscale_image(cache_dir / file_name, input_array[channel_index])
        input_images.append({
            "id": f"input_{channel_index}",
            "label": channel_name,
            "imageUrl": get_cached_image_url(cache_dir, selected_case, file_name)
        })

    return input_images


def save_grayscale_image(file_path, image_array):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (np.clip(image_array, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(image_uint8, mode="L").save(file_path)

    return file_path


def save_error_image(file_path, error_array):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_error = error_array / max(float(np.percentile(error_array, 99.0)), 1e-6)
    normalized_error = np.clip(normalized_error, 0, 1)
    red_channel = (normalized_error * 255).astype(np.uint8)
    green_channel = (np.sqrt(normalized_error) * 180).astype(np.uint8)
    blue_channel = ((1.0 - normalized_error) * 42).astype(np.uint8)
    error_rgb = np.stack([red_channel, green_channel, blue_channel], axis=-1)
    Image.fromarray(error_rgb, mode="RGB").save(file_path)

    return file_path


def get_cached_image_url(cache_dir, selected_case, file_name):
    sample_cache_dir = get_sample_cache_dir(selected_case["split"], selected_case["caseId"])

    if cache_dir == sample_cache_dir:
        return get_sample_image_url(selected_case["split"], selected_case["caseId"], file_name)

    model_id = cache_dir.parts[-4]
    return get_prediction_image_url(model_id, selected_case["split"], selected_case["caseId"], file_name)


def get_optional_checkpoint_info(model_config):
    try:
        return get_checkpoint_info(model_config)
    except HTTPException:
        return None


def get_checkpoint_info(model_config):
    for checkpoint_path in model_config["checkpointCandidates"]:
        if checkpoint_path.exists():
            return {
                "path": str(checkpoint_path),
                "fileName": checkpoint_path.name,
                "modifiedTime": checkpoint_path.stat().st_mtime,
                "sizeBytes": checkpoint_path.stat().st_size
            }

    raise HTTPException(status_code=404, detail=f"{model_config['label']} checkpoint를 찾을 수 없습니다.")


def is_prediction_manifest_ready(manifest_path, checkpoint_info):
    if not manifest_path.exists():
        return False

    manifest = read_json(manifest_path)
    cached_checkpoint = manifest.get("checkpoint", {})

    return (
        cached_checkpoint.get("path") == checkpoint_info["path"] and
        cached_checkpoint.get("modifiedTime") == checkpoint_info["modifiedTime"]
    )


def get_sample_cache_dir(split_name, case_id):
    return RECONSTRUCTION_CACHE_DIR / "samples" / split_name / str(case_id)


def get_prediction_cache_dir(model_id, split_name, case_id):
    return RECONSTRUCTION_CACHE_DIR / model_id / split_name / str(case_id) / "prediction"


def get_sample_image_url(split_name, case_id, file_name):
    return f"{STATIC_RECONSTRUCTION_CACHE_PATH}/samples/{split_name}/{case_id}/{file_name}"


def get_prediction_image_url(model_id, split_name, case_id, file_name):
    return f"{STATIC_RECONSTRUCTION_CACHE_PATH}/{model_id}/{split_name}/{case_id}/prediction/{file_name}"


def add_prediction_image_versions(prediction_manifest, checkpoint_info):
    checkpoint_version = str(int(checkpoint_info["modifiedTime"]))

    for input_channel in prediction_manifest.get("inputChannels", []):
        if "imageUrl" in input_channel:
            input_channel["imageUrl"] = add_url_version(input_channel["imageUrl"], checkpoint_version)

    for image_key in ("target", "prediction", "errorMap"):
        image_manifest = prediction_manifest.get(image_key, {})
        if "imageUrl" in image_manifest:
            image_manifest["imageUrl"] = add_url_version(image_manifest["imageUrl"], checkpoint_version)


def add_url_version(image_url, version):
    base_url = image_url.split("?", 1)[0]

    return f"{base_url}?v={version}"


def clear_cache_dir(cache_dir):
    if not cache_dir.exists():
        return

    for child_path in cache_dir.iterdir():
        if child_path.is_dir():
            clear_cache_dir(child_path)
            child_path.rmdir()
        else:
            child_path.unlink()


def read_json(file_path):
    with file_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json(file_path, payload):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
