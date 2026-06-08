import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU()
        )

    def forward(self, input_tensor):
        return self.block(input_tensor)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(output_channels + skip_channels, output_channels)

    def forward(self, input_tensor, skip_tensor):
        upsampled_tensor = self.up(input_tensor)

        if upsampled_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            upsampled_tensor = F.interpolate(
                upsampled_tensor,
                size=skip_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return self.fuse(torch.cat([upsampled_tensor, skip_tensor], dim=1))


class PlainUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=(32, 64, 128, 256), bottleneck_channels=512):
        super().__init__()
        self.stem = ConvBlock(input_channels, features[0])
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        previous_channels = features[0]

        for feature_channels in features[1:]:
            self.downsamplers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            self.encoder_blocks.append(ConvBlock(previous_channels, feature_channels))
            previous_channels = feature_channels

        self.bottleneck_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(previous_channels, bottleneck_channels)
        decoder_specs = [
            (bottleneck_channels, features[3], features[3]),
            (features[3], features[2], features[2]),
            (features[2], features[1], features[1]),
            (features[1], features[0], features[0])
        ]
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(input_channel_count, skip_channel_count, output_channel_count)
            for input_channel_count, skip_channel_count, output_channel_count in decoder_specs
        ])
        self.output_head = nn.Sequential(
            nn.Conv2d(features[0], output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        skip_tensors = [self.stem(input_tensor)]
        current_tensor = skip_tensors[-1]

        for downsampler, encoder_block in zip(self.downsamplers, self.encoder_blocks):
            current_tensor = encoder_block(downsampler(current_tensor))
            skip_tensors.append(current_tensor)

        current_tensor = self.bottleneck(self.bottleneck_pool(current_tensor))

        for decoder_block, skip_tensor in zip(self.decoder_blocks, reversed(skip_tensors)):
            current_tensor = decoder_block(current_tensor, skip_tensor)

        return self.output_head(current_tensor)


def create_model():
    return PlainUNet()
