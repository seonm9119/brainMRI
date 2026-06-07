from monai.networks.nets import SwinUNETR


def create_model():
    return SwinUNETR(
        in_channels=4,
        out_channels=3,
        patch_size=2,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        feature_size=24,
        norm_name="instance",
        use_checkpoint=True,
        spatial_dims=3
    )
