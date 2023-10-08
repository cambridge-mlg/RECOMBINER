import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor


def count_layer_params(in_dim, out_dim):
    """
    This function counts parameter size for a linear layer
    """
    n_w = in_dim * out_dim
    n_b = out_dim
    return n_w + n_b


def count_net_params(in_dim, hidden_dims, out_dim):
    """
    This function counts parameter size for a given MLP
    """
    dims = [in_dim] + hidden_dims + [out_dim]
    n_params = [count_layer_params(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
    sum_n_params = np.cumsum(n_params)
    return n_params, sum_n_params


def get_training_pair(image_path,
                      feature_size=None,
                      patch=False,
                      patch_size=None):
    """
    Given an image path, this function returns the fourier transformed feature and the rgb values. If patch is True, the
    image will be clipped into patches and this function will return a batch of  fourier transformed feature and the rgb
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

def load_dataset(image_paths, feature_size, patch, patch_size):
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


def PSNR(original, compressed):
    """
    Calculate PSNR of one image, or patches of one image.
    """
    compressed = np.round(np.clip(compressed, 0, 1) * 255) / 255
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr.item()


def batch_PSNR(original, compressed):
    """
    Calculate PSNR of images in one batch
    """
    batch_size = original.shape[0]
    compressed = np.round(np.clip(compressed, 0, 1) * 255) / 255
    mse = np.mean((original.reshape(batch_size, -1) - compressed.reshape(batch_size, -1)) ** 2, axis=-1)
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


"""
Functions below are authored by Zongyu Guo, to generate fourier features as the input
"""


def to_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.
    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non-zero locations of a tensor of ones of
    # same shape as spatial dimensions of image
    coordinates = torch.ones(img.shape[1:]).nonzero(as_tuple=False).float()
    # Normalize coordinates to lie in [-.5, .5]
    coordinates = coordinates / (img.shape[1] - 1) - 0.5
    # Convert to range [-1, 1]
    coordinates *= 2
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features


def make_coord_grid(shape, range, device=None):
    """
        Args:
            shape: tuple
            range: [minv, maxv] or [[minv_1, maxv_1], ..., [minv_d, maxv_d]] for each dim
        Returns:
            grid: shape (*shape, )
    """
    l_lst = []
    for i, s in enumerate(shape):
        l = (0.5 + torch.arange(s, device=device)) / s
        if isinstance(range[0], list) or isinstance(range[0], tuple):
            minv, maxv = range[i]
        else:
            minv, maxv = range
        l = minv + (maxv - minv) * l
        l_lst.append(l)
    grid = torch.meshgrid(*l_lst)
    grid = torch.stack(grid, dim=-1)
    return grid


def to_grid_coordinates_and_features(img):
    """Converts an image to a set of coordinates and features.
    Args:
        img (torch.Tensor): Shape (channels, height, width).
    """
    # Coordinates are indices of all non-zero locations of a tensor of ones of
    # same shape as spatial dimensions of image

    coordinates = make_coord_grid(img.shape[1:], (-1, 1), device=img.device).view(-1, len(img.shape[1:]))
    # Convert image to a tensor of features of shape (num_points, channels)
    features = img.reshape(img.shape[0], -1).T
    return coordinates, features
