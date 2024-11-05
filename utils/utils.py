import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imshows(ims):
    """Visualises a list of dictionaries.

    Each key of the dictionary will be used as a column, and
    each element of the list will be a row.
    """
    nrow = len(ims)
    ncol = len(ims[0])
    fig, axes = plt.subplots(nrow, ncol, figsize=(
        ncol * 3, nrow * 3), facecolor='white')
    for i, im_dict in enumerate(ims):
        for j, (title, im) in enumerate(im_dict.items()):
            if isinstance(im, torch.Tensor):
                im = im.detach().cpu().numpy()
            # If RGB, put to end. Else, average across channel dim
            if im.ndim > 2:
                im = np.moveaxis(im, 0, -1) if im.shape[0] == 3 else np.mean(im, axis=0)

            ax = axes[j] if len(ims) == 1 else axes[i, j]
            ax.set_title(f"{title}\n{im.shape}")
            im_show = ax.imshow(im, cmap='gray')
            ax.axis("off")


def inference(model, data, config, num_data=4):
    from statistics import mean
    to_imshow = []
    mse = []
    with torch.no_grad():
        for idx in np.random.choice(len(data), size=num_data, replace=False):
            rand_data = data[idx]
            rand_input, rand_output_gt = rand_data["input"], rand_data["output"]
            rand_output = model(rand_input.to(config.device)[None])[0]
            to_imshow.append(
                {
                    "FLAIR": rand_input[0],
                    "T1w": rand_input[1],
                    "T2w": rand_input[2],
                    "GT GD": rand_output_gt,
                    "inferred GD": rand_output,
                }
            )
            loss = torch.nn.MSELoss()
            mse.append(loss(rand_output.to("cpu"),rand_output_gt).item())
    imshows(to_imshow)
    print(mean(mse))