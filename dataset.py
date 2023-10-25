"""
Last update on 25-Oct-2023

Process data in different modalities, including images, audio, video and protein
"""
import torch
import numpy as np
import torchvision
from PIL import Image
import pickle
from torchvision.transforms import ToTensor
from torchvision import io, transforms

# 
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

# process video dataset
def process_video_datasets(train_paths, test_paths, save_dir):
    """
    This function processes UCF-101 dataset and store training set and test set into list of tensors.
    Then it save these lists into binary files in the ```save_dir```.
    Args:
        train_paths: List of str. The path of all training instances.
        test_paths: List of str. The path of all test instances.
        save_dir: str. The dir to save binary files.
    """
    def process(path):
        video_tensor = []
        for file_name in path:
            # follow VC-INR paper (https://arxiv.org/abs/2301.09479)
            # for each video, clip out the first 24 frames, center crop by (240, 240) and reshape it to (128, 128) 
            video = io.read_video(file_name)[0].permute([0, 3, 1, 2])[:24, ...]
            if video.shape[-1] >= 240 and video.shape[-2] >= 240:
                video = transforms.CenterCrop([240, 240])(video)
                video = transforms.Resize(size=[128, 128])(video)
                video_tensor.append(video)
        return video_tensor
    train_tensor = process(train_paths)
    test_tensor = process(test_paths)
    with open(save_dir + '/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_tensor, f)
    with open(save_dir + '/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_tensor, f)



def get_ucf_pair(tensor, # T, C(3), W, H
                 feature_size=None,
                 patch=False,
                 patch_size=None):
    tensor = tensor.permute([1, 0, 2, 3])
    c, t, x, y = tensor.shape

    if not patch:
        inputs, outputs = to_grid_coordinates_and_features(tensor)
        w = torch.exp(torch.linspace(0, np.log(1024), feature_size // 4, device=inputs.device))
        inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
        inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
        return inputs, outputs
    else:
        Inputs = []
        Outputs = []
        for x_idx in range(x // patch_size):
            for y_idx in range(y // patch_size):
                patch = tensor[...,
                        x_idx * patch_size: x_idx * patch_size + patch_size,
                        y_idx * patch_size: y_idx * patch_size + patch_size
                        ]
                inputs, outputs = to_grid_coordinates_and_features(patch)
                w = torch.exp(torch.linspace(0, np.log(1024), feature_size // 6, device=inputs.device))
                inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
                inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
                Inputs.append(inputs)
                Outputs.append(outputs)
        Inputs = torch.stack(Inputs)
        Outputs = torch.stack(Outputs)
        return Inputs, Outputs



def load_ucf(video_tensors, feature_size, patch, patch_size):
    # data
    X = []
    Y = []
    for i in video_tensors:
        x, y = get_ucf_pair(i,
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