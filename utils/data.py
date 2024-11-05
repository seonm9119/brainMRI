import os
from glob import glob

from monai.transforms import (
    Compose,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    rescale_array,
    ScaleIntensityd,
)
from monai.data import Dataset, DataLoader, partition_dataset

def load_data(data_dir='.'):
    input_ims = sorted(glob(os.path.join(data_dir, "*input.npy")))
    output_ims = sorted(glob(os.path.join(data_dir, "*GT_output.npy")))

    data = [{"input": i, "output": o} for i, o in zip(input_ims, output_ims)]
    print("number data points", len(data))

    # split data into 80% and 20% for training and validation, respectively
    train_data, val_data = partition_dataset(data, (8, 2), shuffle=True)
    print("num train data points:", len(train_data))
    print("num val data points:", len(val_data))

    return train_data, val_data 


class ChannelWiseScaleIntensityd(MapTransform):
    """Perform channel-wise intensity normalisation."""
    def __init__(self, keys):
        super().__init__(keys)
    def __call__(self, d):
        for key in self.keys:
            for idx, channel in enumerate(d[key]):
                d[key][idx] = rescale_array(channel)
        return d

def build_loader(config):

    keys = ["input", "output"]
    train_transforms = Compose([
        LoadImaged(keys),
        ChannelWiseScaleIntensityd("input"),
        ScaleIntensityd("output"),
        EnsureTyped(keys)
    ])
    
    val_transforms = Compose([
        LoadImaged(keys),
        ChannelWiseScaleIntensityd("input"),
        ScaleIntensityd("output"),
        EnsureTyped(keys),
    ])

    train_data, val_data = load_data(config.data_dir)

    train_loader = DataLoader(Dataset(train_data, train_transforms), 
                          num_workers=config.num_workers, 
                          batch_size=config.batch_size, 
                          shuffle=True)

    val_ds = Dataset(val_data, val_transforms)
    val_loader = DataLoader(val_ds, 
                        num_workers=config.num_workers, 
                        batch_size=config.batch_size, 
                        shuffle=True)
    
    return train_loader, val_ds