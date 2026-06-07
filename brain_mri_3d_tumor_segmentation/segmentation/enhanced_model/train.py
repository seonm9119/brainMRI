import argparse
from pathlib import Path

from brain_mri_3d_tumor_segmentation.segmentation.enhanced_model.model import create_model
from brain_mri_3d_tumor_segmentation.segmentation.training import run_training


SEGMENTATION_ROOT = Path(__file__).resolve().parents[2]
ENHANCED_MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SEGMENTATION_ROOT / "decathlon"
DEFAULT_OUTPUT_DIR = ENHANCED_MODEL_DIR / "checkpoints" / "confidence_swin_unetr"
MODEL_NAME = "confidence_aware_swin_unetr"
MODEL_TITLE = "Confidence-aware SwinUNETR"


def parse_args():
    parser = argparse.ArgumentParser(description="Train confidence-aware SwinUNETR for BRATS segmentation.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--patches-per-case", type=int, default=2)
    parser.add_argument("--sw-batch-size", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-case-limit", type=int)
    parser.add_argument("--val-case-limit", type=int, default=24)
    parser.add_argument("--val-interval", type=int, default=5)
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
