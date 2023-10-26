import sys
# setting path
sys.path.append('../')

import random
import torch
import torchaudio
import numpy as np
import torchvision
from PIL import Image
import pickle
from torchvision.transforms import ToTensor
from torchvision import io, transforms
from utils import to_grid_coordinates_and_features




# LIBRISPEECH class is provided by the authors of COMBINER.
class LIBRISPEECH(torchaudio.datasets.LIBRISPEECH):
    """LIBRISPEECH dataset without labels.

    Args:
        patch_shape (int): Shape of patch to use. If -1, uses all data (no patching).
        num_secs (float): Number of seconds of audio to use. If -1, uses all available
            audio.
        normalize (bool): Whether to normalize data to lie in [0, 1].
    """

    def __init__(
        self,
        patch_shape: int = -1,
        num_secs: float = -1,
        normalize: bool = True,
        *args,
        **kwargs
    ):
        # TODO(emi): We should manually check if root exists, otherwise we should create
        # a directory. Somehow LIBRISPEECH does not do this automatically

        super().__init__(*args, **kwargs)

        # LibriSpeech contains audio 16kHz rate
        self.sample_rate = 16000

        self.normalize = normalize
        self.patch_shape = patch_shape
        self.random_crop = patch_shape != -1
        self.num_secs = num_secs
        self.num_waveform_samples = int(self.num_secs * self.sample_rate)

    def __getitem__(self, index):
        # __getitem__ returns a tuple, where first entry contains raw waveform in [-1, 1]
        datapoint = super().__getitem__(index)[0].float()

        # Normalize data to lie in [0, 1]
        if self.normalize:
            datapoint = (datapoint + 1) / 2

        # Extract only first num_waveform_samples from waveform
        if self.num_secs != -1:
            # Shape (channels, num_waveform_samples)
            datapoint = datapoint[:, : self.num_waveform_samples]

        if self.random_crop:
            datapoint = random_crop1d(datapoint, self.patch_shape)

        return datapoint

def random_crop1d(data, patch_shape: int):
    # print(data.shape)
    # print(data.max(), data.min())
    if not (0 < patch_shape <= data.shape[-1]):
        # data = torch.cat([data, torch.zeros(data.shape[0], patch_shape - data.shape[-1]) + 0.5], dim=1)
        return data
    width_from = random.randint(0, data.shape[-1] - patch_shape)
    return data[
        ...,
        width_from : width_from + patch_shape,
    ]

# process audio data
def process_audio_datasets(save_dir):
    test_dataset = LIBRISPEECH(
            root="./",
            url='test-clean',
            patch_shape=48000,
            num_secs=3,
            download=False,
        )
    test_tensor = list(test_dataset)

    train_dataset = LIBRISPEECH(
        root="./",
        url='train-clean-100',
        patch_shape=48000,
        num_secs=3,
        download=True,
    )
    # the training set is too large. Directly select a subset as the training set when processing
    num = 12000 // 60
    np.random.seed(0)
    idx = np.random.choice(len(train_dataset), num, False)
    np.random.seed(None)
    train_tensor = [train_dataset[i] for i in idx]

    with open(save_dir + '/test_dataset.pkl', 'wb') as f:
        pickle.dump(test_tensor, f)
    with open(save_dir + '/train_dataset.pkl', 'wb') as f:
        pickle.dump(train_tensor, f)


def get_audio_pair(tensor, # C(1), L
                   feature_size=None,
                   patch=False,
                   patch_sizes=None):
    c, x = tensor.shape
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
        for x_idx in range(x // patch_sizes[0]):
            patch = tensor[...,
                            x_idx * patch_sizes[0]: x_idx * patch_sizes[0] + patch_sizes[0],
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

def load_audio(audio_tensors, feature_size, patch, patch_sizes):
    # data
    X = []
    Y = []
    for i in audio_tensors:
        x, y = get_audio_pair(i,
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

