import sys
# setting path
sys.path.append('../')

import torch
import numpy as np
import pickle
from torchvision import io, transforms
from utils import to_grid_coordinates_and_features

# process video dataset
# the parts mapping coordinates to fourier embeddings are modified from COMBINER (https://github.com/cambridge-mlg/combiner)
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
            # for each video, take the first 24 frames, center crop by (240, 240) and reshape it to (128, 128) 
            video = io.read_video(file_name)[0].permute([0, 3, 1, 2])[:24, ...]
            if video.shape[-1] >= 240 and video.shape[-2] >= 240:
                video = transforms.CenterCrop([240, 240])(video)
                video = transforms.Resize(size=[128, 128])(video)
                video_tensor.append(video/255)
        return video_tensor
    train_tensor = process(train_paths)
    test_tensor = process(test_paths)
    with open(save_dir + '/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_tensor, f)
    with open(save_dir + '/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_tensor, f)

def get_video_pair(tensor, # Time, C(3), W, H
                   feature_size=None,
                   patch=False,
                   patch_sizes=None):
    tensor = tensor.permute([1, 0, 2, 3])
    c, t, x, y = tensor.shape
    data_dim = len(tensor.shape) - 1

    if not patch:
        inputs, outputs = to_grid_coordinates_and_features(tensor)
        w = torch.exp(torch.linspace(0, np.log(1024), feature_size // (2*data_dim), device=inputs.device))
        inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
        inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
        return inputs, outputs
    else:
        Inputs = []
        Outputs = []
        for t_idx in range(t // patch_sizes[0]):
            for x_idx in range(x // patch_sizes[1]):
                for y_idx in range(y // patch_sizes[2]):
                    patch = tensor[...,
                                   t_idx * patch_sizes[0]: t_idx * patch_sizes[0] + patch_sizes[0],
                                   x_idx * patch_sizes[1]: x_idx * patch_sizes[1] + patch_sizes[1],
                                   y_idx * patch_sizes[2]: y_idx * patch_sizes[2] + patch_sizes[2]
                                   ]
                    inputs, outputs = to_grid_coordinates_and_features(patch)
                    w = torch.exp(torch.linspace(0, np.log(1024), feature_size // (2*data_dim) , device=inputs.device))
                    inputs = torch.matmul(inputs.unsqueeze(-1), w.unsqueeze(0)).view(*inputs.shape[:-1], -1)
                    inputs = torch.cat([torch.cos(np.pi * inputs), torch.sin(np.pi * inputs)], dim=-1)
                    Inputs.append(inputs)
                    Outputs.append(outputs)
        Inputs = torch.stack(Inputs)
        Outputs = torch.stack(Outputs)
        return Inputs, Outputs

def load_video(video_tensors, feature_size, patch, patch_sizes):
    # data
    X = []
    Y = []
    for i in video_tensors:
        x, y = get_video_pair(i,
                              feature_size=feature_size,
                              patch=patch,
                              patch_sizes=patch_sizes
                              )
        if patch:
            X.append(x)
            Y.append(y)
        else:
            X.append(x[None, ...])
            Y.append(y[None, ...])
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)

