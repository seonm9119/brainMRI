import torch
import torch.nn as nn


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
            nn.SiLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(get_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, image_features):
        return self.layers(image_features)


class AxialSelectiveScan(nn.Module):
    def __init__(self, channel_count):
        super().__init__()
        self.input_projection = nn.Conv2d(channel_count, channel_count * 2, kernel_size=1)
        self.depthwise_conv = nn.Conv2d(channel_count, channel_count, kernel_size=5, padding=2, groups=channel_count)
        self.output_projection = nn.Conv2d(channel_count, channel_count, kernel_size=1)
        self.output_scale = nn.Parameter(torch.ones(1, channel_count, 1, 1) * 0.1)

    def forward(self, image_features):
        scan_source, scan_gate = self.input_projection(image_features).chunk(2, dim=1)
        scan_source = self.depthwise_conv(scan_source)
        context_features = self.scan_height(scan_source) + self.scan_width(scan_source)
        context_features = context_features * 0.5
        gated_context = context_features * torch.sigmoid(scan_gate)

        return self.output_projection(gated_context) * self.output_scale

    def scan_height(self, image_features):
        height = image_features.shape[-2]
        height_steps = torch.arange(1, height + 1, device=image_features.device, dtype=image_features.dtype).view(1, 1, height, 1)
        forward_scan = torch.cumsum(image_features, dim=2) / height_steps
        backward_scan = torch.flip(torch.cumsum(torch.flip(image_features, dims=(2,)), dim=2) / height_steps, dims=(2,))

        return (forward_scan + backward_scan) * 0.5

    def scan_width(self, image_features):
        width = image_features.shape[-1]
        width_steps = torch.arange(1, width + 1, device=image_features.device, dtype=image_features.dtype).view(1, 1, 1, width)
        forward_scan = torch.cumsum(image_features, dim=3) / width_steps
        backward_scan = torch.flip(torch.cumsum(torch.flip(image_features, dims=(3,)), dim=3) / width_steps, dims=(3,))

        return (forward_scan + backward_scan) * 0.5


class MambaConvBlock(nn.Module):
    def __init__(self, channel_count, expansion=2):
        super().__init__()
        hidden_channels = channel_count * expansion
        self.context_norm = nn.GroupNorm(get_group_count(channel_count), channel_count)
        self.context_mixer = AxialSelectiveScan(channel_count)
        self.channel_norm = nn.GroupNorm(get_group_count(channel_count), channel_count)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(channel_count, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channel_count, kernel_size=1)
        )
        self.local_mixer = nn.Sequential(
            nn.Conv2d(channel_count, channel_count, kernel_size=3, padding=1, groups=channel_count, bias=False),
            nn.Conv2d(channel_count, channel_count, kernel_size=1, bias=False),
            nn.GroupNorm(get_group_count(channel_count), channel_count),
            nn.SiLU(inplace=True)
        )

    def forward(self, image_features):
        image_features = image_features + self.context_mixer(self.context_norm(image_features))
        image_features = image_features + self.local_mixer(image_features)
        image_features = image_features + self.channel_mixer(self.channel_norm(image_features))

        return image_features


class SkipFusion(nn.Module):
    def __init__(self, channel_count):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channel_count * 2, channel_count, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, decoder_features, encoder_features):
        fusion_gate = self.gate(torch.cat([decoder_features, encoder_features], dim=1))
        gated_encoder_features = encoder_features * fusion_gate

        return torch.cat([decoder_features, gated_encoder_features], dim=1)


class DecoderStage(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, skip_channels, kernel_size=2, stride=2)
        self.skip_fusion = SkipFusion(skip_channels)
        self.conv = ConvBlock(skip_channels * 2, output_channels)
        self.context = MambaConvBlock(output_channels)

    def forward(self, decoder_features, encoder_features):
        decoder_features = self.upsample(decoder_features)
        fused_features = self.skip_fusion(decoder_features, encoder_features)
        decoded_features = self.conv(fused_features)

        return self.context(decoded_features)


class MambaConvUNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1, features=(32, 64, 128, 256), bottleneck_channels=512):
        super().__init__()
        self.stem = ConvBlock(input_channels, features[0])
        self.encoder_1 = MambaConvBlock(features[0])
        self.down_1 = ConvBlock(features[0], features[1], stride=2)
        self.encoder_2 = MambaConvBlock(features[1])
        self.down_2 = ConvBlock(features[1], features[2], stride=2)
        self.encoder_3 = MambaConvBlock(features[2])
        self.down_3 = ConvBlock(features[2], features[3], stride=2)
        self.encoder_4 = MambaConvBlock(features[3])
        self.down_4 = ConvBlock(features[3], bottleneck_channels, stride=2)
        self.bottleneck = nn.Sequential(
            MambaConvBlock(bottleneck_channels),
            MambaConvBlock(bottleneck_channels)
        )
        self.decoder_4 = DecoderStage(bottleneck_channels, features[3], features[3])
        self.decoder_3 = DecoderStage(features[3], features[2], features[2])
        self.decoder_2 = DecoderStage(features[2], features[1], features[1])
        self.decoder_1 = DecoderStage(features[1], features[0], features[0])
        self.output_head = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(features[0], output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, input_image):
        skip_1 = self.encoder_1(self.stem(input_image))
        skip_2 = self.encoder_2(self.down_1(skip_1))
        skip_3 = self.encoder_3(self.down_2(skip_2))
        skip_4 = self.encoder_4(self.down_3(skip_3))
        bottleneck_features = self.bottleneck(self.down_4(skip_4))
        decoded_features = self.decoder_4(bottleneck_features, skip_4)
        decoded_features = self.decoder_3(decoded_features, skip_3)
        decoded_features = self.decoder_2(decoded_features, skip_2)
        decoded_features = self.decoder_1(decoded_features, skip_1)

        return self.output_head(decoded_features)


def create_model():
    return MambaConvUNet()
