import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm_layer(channel_count):
    return nn.InstanceNorm2d(channel_count, affine=True)


class Pix2PixDownBlock(nn.Module):
    def __init__(self, input_channels, output_channels, use_norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=not use_norm)
        ]

        if use_norm:
            layers.append(get_norm_layer(output_channels))

        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, input_tensor):
        return self.block(input_tensor)


class Pix2PixUpBlock(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm_layer(output_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, input_tensor, skip_tensor=None):
        output_tensor = self.block(input_tensor)

        if skip_tensor is None:
            return output_tensor

        if output_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            output_tensor = F.interpolate(
                output_tensor,
                size=skip_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return torch.cat([output_tensor, skip_tensor], dim=1)


class Pix2PixGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=64):
        super().__init__()
        self.down_1 = Pix2PixDownBlock(input_channels, features, use_norm=False)
        self.down_2 = Pix2PixDownBlock(features, features * 2)
        self.down_3 = Pix2PixDownBlock(features * 2, features * 4)
        self.down_4 = Pix2PixDownBlock(features * 4, features * 8)
        self.down_5 = Pix2PixDownBlock(features * 8, features * 8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1, bias=False),
            get_norm_layer(features * 8),
            nn.ReLU(inplace=True)
        )
        self.up_5 = Pix2PixUpBlock(features * 8, features * 8, dropout=0.25)
        self.up_4 = Pix2PixUpBlock(features * 16, features * 8, dropout=0.25)
        self.up_3 = Pix2PixUpBlock(features * 16, features * 4)
        self.up_2 = Pix2PixUpBlock(features * 8, features * 2)
        self.up_1 = Pix2PixUpBlock(features * 4, features)
        self.output_head = nn.Sequential(
            nn.Conv2d(features * 2, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        skip_1 = self.down_1(input_tensor)
        skip_2 = self.down_2(skip_1)
        skip_3 = self.down_3(skip_2)
        skip_4 = self.down_4(skip_3)
        skip_5 = self.down_5(skip_4)
        bottleneck_tensor = self.bottleneck(skip_5)
        output_tensor = self.up_5(bottleneck_tensor, skip_5)
        output_tensor = self.up_4(output_tensor, skip_4)
        output_tensor = self.up_3(output_tensor, skip_3)
        output_tensor = self.up_2(output_tensor, skip_2)
        output_tensor = self.up_1(output_tensor, skip_1)
        output_tensor = F.interpolate(
            output_tensor,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return self.output_head(output_tensor)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=4, features=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm_layer(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            get_norm_layer(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            get_norm_layer(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input_tensor, target_tensor):
        return self.layers(torch.cat([input_tensor, target_tensor], dim=1))


def create_model():
    return Pix2PixGenerator()


def create_discriminator():
    return PatchDiscriminator()
