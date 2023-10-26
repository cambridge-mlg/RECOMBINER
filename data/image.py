import sys
# setting path
sys.path.append('../')

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor
from utils import to_grid_coordinates_and_features

# process Image dataset
def get_image_pair(image_path,
                   feature_size=None,
                   patch=False,
                   patch_sizes=None):
    image = Image.open(image_path)
    image = ToTensor()(image)
    if image.shape[1] > image.shape[2]: # rotate image to make sure it is landscape layout
        image = image.permute([0, 2, 1])
    c, x, y = image.shape
    data_dim = len(image.shape) - 1

    if not patch:
        inputs, outputs = to_grid_coordinates_and_features(image)
        w = torch.exp(torch.linspace(0, np.log(1024), feature_size // (2*data_dim), device=inputs.device))
        inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
        inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
        return inputs, outputs
    else:
        Inputs = []
        Outputs = []
        for x_idx in range(x // patch_sizes[0]):
            for y_idx in range(y // patch_sizes[1]):
                patch = image[:,
                        x_idx * patch_sizes[0]: x_idx * patch_sizes[0] + patch_sizes[0],
                        y_idx * patch_sizes[1]: y_idx * patch_sizes[1] + patch_sizes[1]
                        ]
                inputs, outputs = to_grid_coordinates_and_features(patch)
                w = torch.exp(torch.linspace(0, np.log(1024), feature_size // (2*data_dim), device=inputs.device))
                inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
                inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
                Inputs.append(inputs)
                Outputs.append(outputs)
        Inputs = torch.stack(Inputs)
        Outputs = torch.stack(Outputs)
        return Inputs, Outputs

def load_image(image_paths, feature_size, patch, patch_sizes):
    # data
    X = []
    Y = []
    for i in image_paths:
        x, y = get_image_pair(i,
                              feature_size=feature_size,
                              patch=patch,
                              patch_size=patch_sizes
                              )
        if patch:
            X.append(x)
            Y.append(y)
        else:
            X.append(x[None, ...])
            Y.append(y[None, ...])
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)

