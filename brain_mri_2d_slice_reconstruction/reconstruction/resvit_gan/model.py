import torch
import torch.nn as nn
import torch.nn.functional as F


def get_group_count(channel_count):
    for group_count in (8, 4, 2):
        if channel_count % group_count == 0:
            return group_count

    return 1


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(get_group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_group_count(output_channels), output_channels),
            nn.GELU()
        )

    def forward(self, input_tensor):
        return self.layers(input_tensor)


class ResidualTransformerBlock(nn.Module):
    def __init__(self, channel_count, heads=8, expansion=4):
        super().__init__()
        self.token_norm = nn.LayerNorm(channel_count)
        self.attention = nn.MultiheadAttention(channel_count, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(channel_count)
        self.mlp = nn.Sequential(
            nn.Linear(channel_count, channel_count * expansion),
            nn.GELU(),
            nn.Linear(channel_count * expansion, channel_count)
        )
        self.local_mixer = nn.Sequential(
            nn.Conv2d(channel_count, channel_count, kernel_size=3, padding=1, groups=channel_count, bias=False),
            nn.Conv2d(channel_count, channel_count, kernel_size=1, bias=False),
            nn.GroupNorm(get_group_count(channel_count), channel_count),
            nn.GELU()
        )
        self.output_scale = nn.Parameter(torch.ones(1, channel_count, 1, 1) * 0.1)

    def forward(self, image_features):
        batch_size, channel_count, height, width = image_features.shape
        tokens = image_features.flatten(2).transpose(1, 2)
        attended_tokens, _ = self.attention(self.token_norm(tokens), self.token_norm(tokens), self.token_norm(tokens), need_weights=False)
        tokens = tokens + attended_tokens
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        transformer_features = tokens.transpose(1, 2).reshape(batch_size, channel_count, height, width)
        mixed_features = transformer_features + self.local_mixer(transformer_features)

        return image_features + mixed_features * self.output_scale


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.fuse = ConvBlock(output_channels + skip_channels, output_channels)

    def forward(self, input_tensor, skip_tensor):
        output_tensor = self.upsample(input_tensor)

        if output_tensor.shape[-2:] != skip_tensor.shape[-2:]:
            output_tensor = F.interpolate(
                output_tensor,
                size=skip_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        return self.fuse(torch.cat([output_tensor, skip_tensor], dim=1))


class ResViTGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=(32, 64, 128, 256), transformer_depth=4):
        super().__init__()
        self.stem = ConvBlock(input_channels, features[0])
        self.down_1 = ConvBlock(features[0], features[1], stride=2)
        self.down_2 = ConvBlock(features[1], features[2], stride=2)
        self.down_3 = ConvBlock(features[2], features[3], stride=2)
        self.down_4 = ConvBlock(features[3], features[3], stride=2)
        self.transformer_bottleneck = nn.Sequential(*[
            ResidualTransformerBlock(features[3], heads=8)
            for _ in range(transformer_depth)
        ])
        self.decoder_4 = DecoderBlock(features[3], features[3], features[3])
        self.decoder_3 = DecoderBlock(features[3], features[2], features[2])
        self.decoder_2 = DecoderBlock(features[2], features[1], features[1])
        self.decoder_1 = DecoderBlock(features[1], features[0], features[0])
        self.output_head = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(features[0], output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_tensor):
        skip_1 = self.stem(input_tensor)
        skip_2 = self.down_1(skip_1)
        skip_3 = self.down_2(skip_2)
        skip_4 = self.down_3(skip_3)
        bottleneck = self.transformer_bottleneck(self.down_4(skip_4))
        output_tensor = self.decoder_4(bottleneck, skip_4)
        output_tensor = self.decoder_3(output_tensor, skip_3)
        output_tensor = self.decoder_2(output_tensor, skip_2)
        output_tensor = self.decoder_1(output_tensor, skip_1)

        return self.output_head(output_tensor)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels=4, features=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(get_group_count(features * 2), features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(get_group_count(features * 4), features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.GroupNorm(get_group_count(features * 8), features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, input_tensor, target_tensor):
        return self.layers(torch.cat([input_tensor, target_tensor], dim=1))


class MultiScalePatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.full_scale = PatchDiscriminator(features=48)
        self.half_scale = PatchDiscriminator(features=48)

    def forward(self, input_tensor, target_tensor):
        full_score = self.full_scale(input_tensor, target_tensor)
        half_input = F.interpolate(input_tensor, scale_factor=0.5, mode="bilinear", align_corners=False)
        half_target = F.interpolate(target_tensor, scale_factor=0.5, mode="bilinear", align_corners=False)
        half_score = self.half_scale(half_input, half_target)

        return [full_score, half_score]


def create_model():
    return ResViTGenerator()


def create_discriminator():
    return MultiScalePatchDiscriminator()
