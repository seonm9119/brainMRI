import argparse
from pathlib import Path

from brain_mri_3d_tumor_segmentation.segmentation.training import run_training
from .model import create_model


SEGMENTATION_ROOT = Path(__file__).resolve().parents[2]
ASSIGNMENT_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SEGMENTATION_ROOT / "decathlon"
DEFAULT_OUTPUT_DIR = ASSIGNMENT_MODEL_DIR / "checkpoints" / "assignment_unet"
MODEL_NAME = "assignment_unet3d_baseline"
MODEL_TITLE = "3D U-Net Baseline"


def parse_args():
    parser = argparse.ArgumentParser(description="Train the assignment baseline 3D U-Net for BRATS segmentation.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(128, 128, 128))
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-case-limit", type=int)
    parser.add_argument("--val-case-limit", type=int, default=16)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--cache-rate", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args, Path(args.output_dir), create_model, MODEL_NAME, MODEL_TITLE)


if __name__ == "__main__":
    main()
