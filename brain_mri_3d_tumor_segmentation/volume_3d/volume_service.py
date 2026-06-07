import json
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import HTTPException


SEGMENTATION_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = SEGMENTATION_ROOT / "sample"
SELECTED_CASES_PATH = SAMPLE_DIR / "selected_cases.json"
NIFTI_CACHE_DIR = SEGMENTATION_ROOT / "nifti_cache"
STATIC_NIFTI_CACHE_PATH = "/static/nifti-cache"
INPUT_VOLUME_FIT_RENDER_SCALE = 0.72


def get_selected_cases_response():
    selected_cases = load_selected_cases()

    return {
        "cases": selected_cases["cases"],
        "modalities": selected_cases["modalities"]
    }


def get_modality_nifti_response(case_id, modality):
    selected_cases = load_selected_cases()
    normalized_case_id = normalize_case_id(case_id)
    normalized_modality = modality.lower()
    selected_case = find_selected_case(selected_cases, normalized_case_id)
    selected_modality = find_selected_modality(selected_cases, normalized_modality)
    cache_dir = NIFTI_CACHE_DIR / normalized_case_id
    metadata_path = cache_dir / f"{normalized_modality}.json"
    nifti_path = cache_dir / f"{normalized_modality}.nii.gz"

    if metadata_path.exists() and nifti_path.exists():
        metadata = read_json(metadata_path)
        metadata["cacheHit"] = True
        metadata["niftiUrl"] = get_nifti_url(normalized_case_id, normalized_modality)
        metadata["renderSettings"] = get_render_settings(normalized_modality, metadata["intensityRange"])
        return metadata

    case_path = get_case_path(selected_case)
    cache_dir.mkdir(parents=True, exist_ok=True)

    volume, original_shape, affine, header = load_nifti_modality_with_header(case_path, selected_modality)
    clean_volume, intensity_range = prepare_nifti_volume(volume)
    save_modality_nifti(nifti_path, clean_volume, affine, header, intensity_range)

    metadata = {
        "caseId": selected_case["caseId"],
        "fileName": selected_case["fileName"],
        "modality": selected_modality["id"],
        "modalityLabel": selected_modality["label"],
        "channelIndex": selected_modality["channelIndex"],
        "originalShape": original_shape,
        "shape": list(clean_volume.shape),
        "dtype": "float32",
        "niftiUrl": get_nifti_url(normalized_case_id, normalized_modality),
        "intensityRange": intensity_range,
        "renderSettings": get_render_settings(selected_modality["id"], intensity_range)
    }
    write_json(metadata_path, metadata)

    metadata["cacheHit"] = False
    return metadata


def load_selected_cases():
    if not SELECTED_CASES_PATH.exists():
        raise HTTPException(status_code=500, detail="selected_cases.json 파일을 찾을 수 없습니다.")

    return read_json(SELECTED_CASES_PATH)


def normalize_case_id(case_id):
    return case_id.removesuffix(".nii.gz")


def find_selected_case(selected_cases, case_id):
    for selected_case in selected_cases["cases"]:
        if selected_case["caseId"] == case_id:
            return selected_case

    raise HTTPException(status_code=404, detail=f"{case_id} 샘플을 찾을 수 없습니다.")


def find_selected_modality(selected_cases, modality):
    for selected_modality in selected_cases["modalities"]:
        if selected_modality["id"].lower() == modality:
            return selected_modality

    raise HTTPException(status_code=404, detail=f"{modality} modality를 찾을 수 없습니다.")


def get_case_path(selected_case):
    case_path = SEGMENTATION_ROOT / selected_case["relativePath"]

    if not case_path.exists():
        raise HTTPException(status_code=404, detail=f"{selected_case['fileName']} 파일을 찾을 수 없습니다.")

    return case_path


def load_nifti_modality_with_header(case_path, selected_modality):
    nifti_image = nib.load(str(case_path))
    original_shape = list(nifti_image.shape)
    channel_index = selected_modality["channelIndex"]

    if len(original_shape) != 4:
        raise HTTPException(status_code=422, detail="4D NIfTI volume만 modality NIfTI로 변환할 수 있습니다.")

    if channel_index >= original_shape[3]:
        raise HTTPException(status_code=422, detail="요청한 modality channel이 NIfTI shape 범위를 벗어났습니다.")

    volume = np.asarray(nifti_image.dataobj[:, :, :, channel_index], dtype=np.float32)
    return volume, original_shape, nifti_image.affine, nifti_image.header.copy()


def prepare_nifti_volume(volume):
    clean_volume = np.nan_to_num(volume, nan=0, posinf=0, neginf=0).astype(np.float32)
    foreground = clean_volume[clean_volume > 0]

    if foreground.size:
        lower, upper = np.percentile(foreground, [0.5, 99.7])
    else:
        lower = float(clean_volume.min())
        upper = float(clean_volume.max())

    if upper <= lower:
        upper = lower + 1

    return clean_volume, {
        "lower": float(lower),
        "upper": float(upper)
    }


def save_modality_nifti(nifti_path, volume, affine, header, intensity_range):
    header.set_data_shape(volume.shape)
    header.set_data_dtype(np.float32)
    header["cal_min"] = intensity_range["lower"]
    header["cal_max"] = intensity_range["upper"]

    nifti_image = nib.Nifti1Image(volume, affine, header)
    nib.save(nifti_image, str(nifti_path))


def get_render_settings(modality, intensity_range):
    lower = intensity_range["lower"]
    upper = intensity_range["upper"]
    window = upper - lower

    modality_settings = {
        "flair": {
            "calMin": lower + (window * 0.08),
            "calMax": lower + (window * 0.96),
            "opacity": 1,
            "colormap": "gray",
            "azimuth": 136,
            "elevation": 20,
            "scale": INPUT_VOLUME_FIT_RENDER_SCALE
        },
        "t1w": {
            "calMin": lower + (window * 0.04),
            "calMax": lower + (window * 0.90),
            "opacity": 1,
            "colormap": "gray",
            "azimuth": 134,
            "elevation": 18,
            "scale": INPUT_VOLUME_FIT_RENDER_SCALE
        },
        "t1gd": {
            "calMin": lower + (window * 0.12),
            "calMax": upper,
            "opacity": 1,
            "colormap": "gray",
            "azimuth": 136,
            "elevation": 22,
            "scale": INPUT_VOLUME_FIT_RENDER_SCALE
        },
        "t2w": {
            "calMin": lower + (window * 0.08),
            "calMax": lower + (window * 0.97),
            "opacity": 1,
            "colormap": "gray",
            "azimuth": 136,
            "elevation": 20,
            "scale": INPUT_VOLUME_FIT_RENDER_SCALE
        }
    }

    return modality_settings.get(modality, modality_settings["flair"])


def get_nifti_url(case_id, modality):
    return f"{STATIC_NIFTI_CACHE_PATH}/{case_id}/{modality}.nii.gz"


def read_json(file_path):
    with file_path.open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


def write_json(file_path, content):
    with file_path.open("w", encoding="utf-8") as json_file:
        json.dump(content, json_file, ensure_ascii=False, indent=2)
