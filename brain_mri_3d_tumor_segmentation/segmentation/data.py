import json
from pathlib import Path

import numpy as np
from monai.transforms import MapTransform


REGION_NAMES = ("TC", "WT", "ET")


def load_decathlon_cases(data_dir):
    data_dir = Path(data_dir)
    dataset_path = data_dir / "dataset.json"

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        dataset_config = json.load(dataset_file)

    cases = []

    for training_case in dataset_config["training"]:
        image_path = data_dir / training_case["image"].removeprefix("./")
        label_path = data_dir / training_case["label"].removeprefix("./")
        case_id = image_path.name.removesuffix(".nii.gz")
        cases.append({
            "caseId": case_id,
            "imagePath": image_path,
            "labelPath": label_path
        })

    return cases


def split_cases(cases, val_ratio, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(cases))
    rng.shuffle(indices)
    val_count = max(1, int(len(cases) * val_ratio))
    val_indices = set(indices[:val_count].tolist())
    train_cases = [case for case_index, case in enumerate(cases) if case_index not in val_indices]
    val_cases = [case for case_index, case in enumerate(cases) if case_index in val_indices]

    return train_cases, val_cases


def to_monai_records(cases):
    return [
        {
            "caseId": selected_case["caseId"],
            "image": str(selected_case["imagePath"]),
            "label": str(selected_case["labelPath"])
        }
        for selected_case in cases
    ]


class ConvertBratsLabelToRegionsd(MapTransform):
    def __call__(self, case_data):
        transformed_case_data = dict(case_data)

        for key in self.keys:
            label = np.asarray(transformed_case_data[key])

            if label.shape[0] == 1:
                label = label[0]

            tumor_core = np.logical_or(label == 2, label == 3)
            whole_tumor = label > 0
            enhancing_tumor = label == 3
            transformed_case_data[key] = np.stack(
                [tumor_core, whole_tumor, enhancing_tumor],
                axis=0
            ).astype(np.float32)

        return transformed_case_data
