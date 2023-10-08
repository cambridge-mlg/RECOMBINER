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




def PSNR(original, compressed, round):
    """
    Calculate PSNR of one image, or patches of one image.
    """
    if round:
        compressed = np.round(np.clip(compressed, 0, 1) * 255) / 255
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr.item()


def batch_PSNR(original, compressed, round):
    """
    Calculate PSNR of images in one batch
    """
    batch_size = original.shape[0]
    if round:
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
