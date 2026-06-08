import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_NAMES = ("train", "val", "test")
INPUT_SUFFIX = "_input.npy"
TARGET_SUFFIX = "_GT_output.npy"
INPUT_CHANNELS = ("FLAIR", "T1w", "T2w")
TARGET_CHANNEL = "T1Gd"


def validate_split_name(split_name):
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"지원하지 않는 split입니다: {split_name}")


def load_reconstruction_cases(data_dir, split_name):
    validate_split_name(split_name)
    split_dir = Path(data_dir) / split_name

    if not split_dir.exists():
        raise FileNotFoundError(f"reconstruction split 폴더를 찾을 수 없습니다: {split_dir}")

    input_paths = {
        input_path.name.removesuffix(INPUT_SUFFIX): input_path
        for input_path in split_dir.glob(f"*{INPUT_SUFFIX}")
    }
    target_paths = {
        target_path.name.removesuffix(TARGET_SUFFIX): target_path
        for target_path in split_dir.glob(f"*{TARGET_SUFFIX}")
    }
    case_ids = sorted(input_paths.keys() & target_paths.keys(), key=sort_case_id)

    return [
        {
            "caseId": case_id,
            "split": split_name,
            "inputPath": input_paths[case_id],
            "targetPath": target_paths[case_id]
        }
        for case_id in case_ids
    ]


def get_reconstruction_case(data_dir, split_name, case_id):
    normalized_case_id = normalize_case_id(case_id)
    matching_cases = [
        selected_case
        for selected_case in load_reconstruction_cases(data_dir, split_name)
        if normalize_case_id(selected_case["caseId"]) == normalized_case_id
    ]

    if not matching_cases:
        raise FileNotFoundError(f"{split_name} split에서 reconstruction case를 찾을 수 없습니다: {case_id}")

    return matching_cases[0]


def normalize_case_id(case_id):
    return str(case_id).removesuffix(INPUT_SUFFIX).removesuffix(TARGET_SUFFIX)


def sort_case_id(case_id):
    return int(case_id) if str(case_id).isdigit() else str(case_id)


def load_case_arrays(selected_case):
    input_array = np.load(selected_case["inputPath"]).astype(np.float32)
    target_array = np.load(selected_case["targetPath"]).astype(np.float32)

    return normalize_input_array(input_array), normalize_target_array(target_array)


def normalize_input_array(input_array):
    return np.stack([scale_image_to_unit(channel) for channel in input_array], axis=0).astype(np.float32)


def normalize_target_array(target_array):
    if target_array.ndim == 2:
        target_array = target_array[None, ...]

    return np.stack([scale_image_to_unit(channel) for channel in target_array], axis=0).astype(np.float32)


def scale_image_to_unit(image_array):
    finite_mask = np.isfinite(image_array)

    if not finite_mask.any():
        return np.zeros_like(image_array, dtype=np.float32)

    finite_values = image_array[finite_mask]
    lower_bound = np.percentile(finite_values, 0.5)
    upper_bound = np.percentile(finite_values, 99.5)

    if upper_bound - lower_bound < 1e-6:
        lower_bound = float(finite_values.min())
        upper_bound = float(finite_values.max())

    if upper_bound - lower_bound < 1e-6:
        return np.zeros_like(image_array, dtype=np.float32)

    scaled_image = (image_array - lower_bound) / (upper_bound - lower_bound)
    return np.clip(scaled_image, 0.0, 1.0).astype(np.float32)


class Brain2DSliceDataset(Dataset):
    def __init__(self, data_dir, split_name, augment=False, case_limit=None):
        self.cases = load_reconstruction_cases(data_dir, split_name)
        self.augment = augment

        if case_limit:
            self.cases = self.cases[:case_limit]

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, case_index):
        selected_case = self.cases[case_index]
        input_array, target_array = load_case_arrays(selected_case)

        if self.augment:
            input_array, target_array = augment_case_arrays(input_array, target_array)

        return {
            "caseId": selected_case["caseId"],
            "input": torch.from_numpy(np.ascontiguousarray(input_array)),
            "target": torch.from_numpy(np.ascontiguousarray(target_array))
        }


def augment_case_arrays(input_array, target_array):
    if random.random() < 0.5:
        input_array = np.flip(input_array, axis=2)
        target_array = np.flip(target_array, axis=2)

    if random.random() < 0.5:
        input_array = np.flip(input_array, axis=1)
        target_array = np.flip(target_array, axis=1)

    if random.random() < 0.2:
        rotation_count = random.randint(1, 3)
        input_array = np.rot90(input_array, k=rotation_count, axes=(1, 2))
        target_array = np.rot90(target_array, k=rotation_count, axes=(1, 2))

    return input_array.copy(), target_array.copy()
