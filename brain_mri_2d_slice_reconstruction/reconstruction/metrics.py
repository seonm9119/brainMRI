import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_ssim(prediction, target, window_size=7):
    padding = window_size // 2
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    prediction_mean = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    target_mean = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    prediction_variance = F.avg_pool2d(prediction * prediction, window_size, stride=1, padding=padding) - prediction_mean ** 2
    target_variance = F.avg_pool2d(target * target, window_size, stride=1, padding=padding) - target_mean ** 2
    covariance = F.avg_pool2d(prediction * target, window_size, stride=1, padding=padding) - prediction_mean * target_mean

    numerator = (2 * prediction_mean * target_mean + c1) * (2 * covariance + c2)
    denominator = (prediction_mean ** 2 + target_mean ** 2 + c1) * (prediction_variance + target_variance + c2)
    return (numerator / denominator.clamp_min(1e-6)).mean()


def calculate_gradient_loss(prediction, target):
    prediction_dx, prediction_dy = calculate_image_gradients(prediction)
    target_dx, target_dy = calculate_image_gradients(target)
    return F.l1_loss(prediction_dx, target_dx) + F.l1_loss(prediction_dy, target_dy)


def calculate_weighted_gradient_loss(prediction, target, pixel_weights):
    prediction_dx, prediction_dy = calculate_image_gradients(prediction)
    target_dx, target_dy = calculate_image_gradients(target)

    return (
        calculate_weighted_l1_loss(prediction_dx, target_dx, pixel_weights) +
        calculate_weighted_l1_loss(prediction_dy, target_dy, pixel_weights)
    )


def calculate_weighted_l1_loss(prediction, target, pixel_weights):
    loss_map = torch.abs(prediction - target)

    return (loss_map * pixel_weights).sum() / pixel_weights.sum().clamp_min(1e-6)


def create_foreground_mask(target, threshold=0.02):
    foreground_mask = (target > threshold).float()
    foreground_mask = F.max_pool2d(foreground_mask, kernel_size=3, stride=1, padding=1)

    return foreground_mask


def create_foreground_weights(target, foreground_weight=2.0, background_weight=0.25, threshold=0.02):
    foreground_mask = create_foreground_mask(target, threshold)

    return foreground_mask * foreground_weight + (1.0 - foreground_mask) * background_weight


def calculate_image_gradients(image_tensor):
    dx = image_tensor[..., :, 1:] - image_tensor[..., :, :-1]
    dy = image_tensor[..., 1:, :] - image_tensor[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))

    return dx, dy


def calculate_reconstruction_metrics(prediction, target):
    prediction = prediction.clamp(0, 1)
    target = target.clamp(0, 1)
    mae = F.l1_loss(prediction, target).item()
    mse = F.mse_loss(prediction, target).item()
    rmse = math.sqrt(max(mse, 0.0))
    psnr = 99.0 if mse <= 1e-12 else 20 * math.log10(1.0 / math.sqrt(mse))
    ssim = calculate_ssim(prediction, target).item()
    foreground_metrics = calculate_foreground_metrics(prediction, target)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "psnr": float(psnr),
        "ssim": float(ssim),
        **foreground_metrics
    }


def calculate_foreground_metrics(prediction, target):
    foreground_mask = create_foreground_mask(target).bool()

    if not foreground_mask.any():
        return {
            "foregroundMae": 0.0,
            "foregroundRmse": 0.0,
            "foregroundPsnr": 99.0
        }

    foreground_prediction = prediction[foreground_mask]
    foreground_target = target[foreground_mask]
    foreground_mae = F.l1_loss(foreground_prediction, foreground_target).item()
    foreground_mse = F.mse_loss(foreground_prediction, foreground_target).item()
    foreground_rmse = math.sqrt(max(foreground_mse, 0.0))
    foreground_psnr = 99.0 if foreground_mse <= 1e-12 else 20 * math.log10(1.0 / math.sqrt(foreground_mse))

    return {
        "foregroundMae": float(foreground_mae),
        "foregroundRmse": float(foreground_rmse),
        "foregroundPsnr": float(foreground_psnr)
    }


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        l1_weight=1.0,
        ssim_weight=0.35,
        gradient_weight=0.08,
        foreground_weight=0.0,
        background_weight=0.25,
        foreground_threshold=0.02
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.foreground_weight = foreground_weight
        self.background_weight = background_weight
        self.foreground_threshold = foreground_threshold

    def forward(self, prediction, target):
        if self.foreground_weight > 0:
            pixel_weights = create_foreground_weights(target, self.foreground_weight, self.background_weight, self.foreground_threshold)
            l1_loss = calculate_weighted_l1_loss(prediction, target, pixel_weights)
            gradient_loss = calculate_weighted_gradient_loss(prediction, target, pixel_weights)
        else:
            l1_loss = F.l1_loss(prediction, target)
            gradient_loss = calculate_gradient_loss(prediction, target)

        ssim_loss = 1.0 - calculate_ssim(prediction, target)
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss + self.gradient_weight * gradient_loss

        return total_loss, {
            "l1Loss": float(l1_loss.detach().cpu()),
            "ssimLoss": float(ssim_loss.detach().cpu()),
            "gradientLoss": float(gradient_loss.detach().cpu()),
            "totalLoss": float(total_loss.detach().cpu())
        }
