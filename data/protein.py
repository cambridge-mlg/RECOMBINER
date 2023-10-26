import numpy as np
import pickle
import torch
import pickle
from utils import to_grid_coordinates_and_features


# process protein PDB files
def process_protein_datasets(train_pdb_paths, test_pdb_paths, save_dir):

    def _process(pdb_paths):
        MAX_LENGTH = 96

        backbone_clips = []
        for i in pdb_paths:
            with open(i, 'r') as f:
                clips = []
                f = f.readlines()
                l = 0
                try:
                    for line in f:
                        if line.split()[0] == 'ATOM':
                            if line.split()[2] in ['CA']:
                                l += 1
                                if len(clips) < MAX_LENGTH:
                                    clips.append([float(i) for i in line.split()[6:9]])
                except:
                    pass
                if l >= MAX_LENGTH:
                    clips = clips[0:MAX_LENGTH]
                    backbone_clips.append(clips)
        normalized_backbone_clips = [((torch.tensor(backbone_clips[i]) - torch.tensor(backbone_clips[i]).mean(0))/25).T for i in range(len(backbone_clips))]
        # Note, that we normalize the xyz structures by 25. Therefore, when calculating RMSD, we should scale 25 back!
        return normalized_backbone_clips
    
    train_dataset = _process(train_pdb_paths)
    test_dataset = _process(test_pdb_paths)

    with open(save_dir + '/train_dataset.pkl', 'wb') as f_out:
        pickle.dump(train_dataset, f_out) 
    with open(save_dir + '/test_dataset.pkl', 'wb') as f_out:
        pickle.dump(test_dataset, f_out) 


def get_protein_pair(tensor, # C(3), L
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

def load_protein(protein_tensors, feature_size, patch, patch_sizes):
    # data
    X = []
    Y = []
    for i in protein_tensors:
        x, y = get_protein_pair(i,
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


