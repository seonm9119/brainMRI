import argparse
from pathlib import Path

from brain_mri_2d_slice_reconstruction.reconstruction.gan_training import run_gan_training
from .model import create_discriminator, create_model


RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = RECONSTRUCTION_ROOT / "brain_2d"
DEFAULT_OUTPUT_DIR = MODEL_DIR / "checkpoints" / "plain_gan"
MODEL_NAME = "plain_gan"
MODEL_TITLE = "Plain cGAN Pix2Pix Baseline"


def parse_args():
    parser = argparse.ArgumentParser(description="Train plain Pix2Pix cGAN for 2D brain MRI synthesis.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-epochs", "--epochs", dest="max_epochs", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--generator-learning-rate", type=float, default=2e-4)
    parser.add_argument("--discriminator-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adversarial-weight", type=float, default=0.02)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--ssim-weight", type=float, default=0.35)
    parser.add_argument("--gradient-weight", type=float, default=0.08)
    parser.add_argument("--foreground-weight", type=float, default=0.0)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--foreground-threshold", type=float, default=0.02)
    parser.add_argument("--train-case-limit", type=int)
    parser.add_argument("--val-case-limit", type=int)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--val-loss-patience", type=int, default=12)
    parser.add_argument("--metric-patience", type=int, default=20)
    parser.add_argument("--val-loss-min-delta", type=float, default=1e-4)
    parser.add_argument("--metric-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-warmup", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialize-from")
    parser.add_argument("--no-amp", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    run_gan_training(args, Path(args.output_dir), create_model, create_discriminator, MODEL_NAME, MODEL_TITLE)


if __name__ == "__main__":
    main()
