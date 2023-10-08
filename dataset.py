"""
This file processes the dataset for image, audio, video and protein.


Author: Jiajun He (jh2383@cam.ac.uk)
Last Update: 2023-10-8
"""

import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor


# image, including cifar / kodak
def get_training_pair_image(image_path,
                            feature_size=None,
                            patch=False,
                            patch_size=None):
    """
    Given an image path, this function returns the fourier transformed feature and the rgb values. If patch is True, the
    image will be cropped into patches and this function will return a batch of fourier transformed feature and the rgb
    values.
    """
    image = Image.open(image_path)
    image = ToTensor()(image)
    if image.shape[1] > image.shape[2]: # rotate image to make sure it is landscape layout
        image = image.permute([0, 2, 1])
    c, x, y = image.shape

    if not patch:
        inputs, outputs = to_grid_coordinates_and_features(image)
        w = torch.exp(torch.linspace(0, np.log(1024), feature_size // 4, device=inputs.device))
        inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
        inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
        return inputs, outputs
    else:
        Inputs = []
        Outputs = []
        for x_idx in range(x // patch_size):
            for y_idx in range(y // patch_size):
                patch = image[:,
                        x_idx * patch_size: x_idx * patch_size + patch_size,
                        y_idx * patch_size: y_idx * patch_size + patch_size
                        ]
                inputs, outputs = to_grid_coordinates_and_features(patch)
                w = torch.exp(torch.linspace(0, np.log(1024), feature_size // 4, device=inputs.device))
                inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
                inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
                Inputs.append(inputs)
                Outputs.append(outputs)
        Inputs = torch.stack(Inputs)
        Outputs = torch.stack(Outputs)
        return Inputs, Outputs
def load_dataset_image(image_paths, feature_size, patch, patch_size):
    # data
    X = []
    Y = []
    for i in image_paths:
        x, y = get_training_pair(i,
                                 feature_size=feature_size,
                                 patch=patch,
                                 patch_size=patch_size
                                 )
        if patch:
            X.append(x)
            Y.append(y)
        else:
            X.append(x[None, :, :])
            Y.append(y[None, :, :])
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)