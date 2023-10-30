import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import kl_divergence, Normal

import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

import pickle
from tqdm import tqdm

from utils import load_dataset, PSNR, batch_PSNR, count_net_params

from torch.quasirandom import SobolEngine
from scipy.stats import norm



class Sine(nn.Module):
    """Sine activation with scaling.
    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class TestBNNmodel(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dims,
                 out_dim,
                 number_of_datapoints,
                 upsample_factor,
                 latent_dim,
                 mix_datapoints, # True when trained on patches from one images from Kodak
                 pixel_size,
                 patch, # True if on Kodak
                 patch_size=32,
                 dataset='kodak',
                 param_mapping=None,
                 coord_mapping=None,
                 random_seed=42,
                 w0=30.,
                 c=6.,
                 eps_beta_0=1e-8,
                 eps_beta=5e-5,
                 device="cpu",
                 n_pixels=512*768,
                 p_loc=None,
                 p_log_scale=None,
                 init_log_scale=-4.,
                 param_to_group=None,
                 group_to_param=None,
                 n_groups=None,
                 group_start_index=None,
                 group_end_index=None,
                 group_idx=None, # indicate the group of each position in self.loc (if default and correct executed, it should be 0, 0, ..., 0, 1, .., 1, 2, ..., 2, ...)
                 h_p_loc=None,
                 h_p_log_scale=None,
                 h_init_log_scale=-4.,
                 h_param_to_group=None,
                 h_group_to_param=None,
                 h_n_groups=None,
                 h_group_start_index=None,
                 h_group_end_index=None,
                 h_group_idx=None,
                 hh_p_loc=None,
                 hh_p_log_scale=None,
                 hh_init_log_scale=-4.,
                 hh_param_to_group=None,
                 hh_group_to_param=None,
                 hh_n_groups=None,
                 hh_group_start_index=None,
                 hh_group_end_index=None,
                 hh_group_idx=None,
                 search_space_size=32,
                 kl_buffer=0.2,
                 kl_adjust_epoch=2000,
                 kl_adjust_gap=10,
                 kl_boundary=0.2,
                 ):
        super().__init__()
        self.bit_per_group = 16
        self.n_layers = len(hidden_dims) + 1
        self.dims = [in_dim] + hidden_dims + [out_dim]
        self.device = device
        self.dataset = dataset
        self.random_seed = random_seed
        self.st = lambda x: F.softplus(x, beta=1, threshold=20) / 6

        self.upsample_factor = upsample_factor
        self.latent_dim = latent_dim

        self.patch = patch
        self.patch_size = patch_size
        self.mix_datapoints = mix_datapoints
        self.pixel_size = pixel_size

        self.param_mapping = param_mapping
        self.coord_mapping = coord_mapping

        try:
            for param in self.param_mapping.parameters():
                param.requires_grad = False
        except:
            pass
        try:
            for param in self.coord_mapping.parameters():
                param.requires_grad = False
        except:
            pass

        # calculate number of parameters
        _, self.cum_param_sizes = count_net_params(in_dim, hidden_dims, out_dim)

        # parameter and its grouping
        self.param_to_group = param_to_group
        self.group_to_param = group_to_param
        self.n_groups = n_groups
        self.group_start_index = group_start_index
        self.group_end_index = group_end_index
        self.group_idx = group_idx.astype(int)
        number_of_params = p_loc.shape[0]
        loc_data = p_loc[None, :].repeat([number_of_datapoints, 1]).to(device)
        self.log_scale = nn.Parameter(torch.zeros([number_of_datapoints, number_of_params]) + init_log_scale)
        self.loc = nn.Parameter(loc_data.clone())

        # parameter and its grouping
        self.h_param_to_group = h_param_to_group
        self.h_group_to_param = h_group_to_param
        self.h_n_groups = h_n_groups
        self.h_group_start_index = h_group_start_index
        self.h_group_end_index = h_group_end_index
        self.h_group_idx = h_group_idx.astype(int)
        h_number_of_params = h_p_loc.shape[0]
        h_loc_data = h_p_loc[None, :].repeat([number_of_datapoints//16, 1]).to(device)
        self.h_log_scale = nn.Parameter(torch.zeros([number_of_datapoints//16, h_number_of_params]) + h_init_log_scale)
        self.h_loc = nn.Parameter(h_loc_data.clone())

        # parameter and its grouping
        self.hh_param_to_group = hh_param_to_group
        self.hh_group_to_param = hh_group_to_param
        self.hh_n_groups = hh_n_groups
        self.hh_group_start_index = hh_group_start_index
        self.hh_group_end_index = hh_group_end_index
        self.hh_group_idx = hh_group_idx.astype(int)
        hh_number_of_params = hh_p_loc.shape[0]
        hh_loc_data = hh_p_loc[None, :].repeat([number_of_datapoints//96, 1]).to(device)
        self.hh_log_scale = nn.Parameter(torch.zeros([number_of_datapoints//96, hh_number_of_params]) + hh_init_log_scale)
        self.hh_loc = nn.Parameter(hh_loc_data.clone())


        # KL scale factor of each group
        self.kl_beta = torch.zeros([number_of_datapoints, self.n_groups]) + eps_beta_0
        self.h_kl_beta = torch.zeros([number_of_datapoints//16, self.h_n_groups]) + eps_beta_0
        self.hh_kl_beta = torch.zeros([number_of_datapoints//96, self.hh_n_groups]) + eps_beta_0

        self.eps_beta = eps_beta # adjust step size
        self.kl_buffer = kl_buffer # buffer (the range in which beta is not updated)
        self.kl_boundary = kl_boundary # boundary (how far to keep from the budget)
        self.kl_adjust_epoch = kl_adjust_epoch # after which the model start to adjust budget
        self.kl_adjust_gap = kl_adjust_gap

        # if patching, randomly permute the columns to allocate budgets better
        if self.patch:
            self.permute_patch_list_g2p = []  # for each dimension of loc/scale (colmun), permute them
            self.permute_patch_list_p2g = []
            for dim_idx in range(self.loc.shape[1]):
                if mix_datapoints:
                    np.random.seed(dim_idx)
                    patch_order = np.random.choice(self.loc.shape[0], self.loc.shape[0], False)
                    self.permute_patch_list_g2p.append(patch_order)
                    self.permute_patch_list_p2g.append(np.argsort(patch_order))
                    np.random.seed(None)
                else:
                    self.permute_patch_list_g2p.append(np.arange(self.loc.shape[0]))
                    self.permute_patch_list_p2g.append(np.arange(self.loc.shape[0]))
            self.permute_patch_x_g2p = np.vstack(self.permute_patch_list_g2p).T
            self.permute_patch_y_g2p = torch.arange(self.loc.shape[1])[None, :].repeat([self.loc.shape[0], 1])
            self.permute_patch_x_p2g = np.vstack(self.permute_patch_list_p2g).T
            self.permute_patch_y_p2g = torch.arange(self.loc.shape[1])[None, :].repeat([self.loc.shape[0], 1])

            self.h_permute_patch_list_g2p = []  # for each dimension of loc/scale (colmun), permute them
            self.h_permute_patch_list_p2g = []
            for dim_idx in range(self.h_loc.shape[1]):
                if mix_datapoints:
                    np.random.seed(dim_idx)
                    patch_order = np.random.choice(self.h_loc.shape[0], self.h_loc.shape[0], False)
                    self.h_permute_patch_list_g2p.append(patch_order)
                    self.h_permute_patch_list_p2g.append(np.argsort(patch_order))
                    np.random.seed(None)
                else:
                    self.h_permute_patch_list_g2p.append(np.arange(self.h_loc.shape[0]))
                    self.h_permute_patch_list_p2g.append(np.arange(self.h_loc.shape[0]))
            self.h_permute_patch_x_g2p = np.vstack(self.h_permute_patch_list_g2p).T
            self.h_permute_patch_y_g2p = torch.arange(self.h_loc.shape[1])[None, :].repeat([self.h_loc.shape[0], 1])
            self.h_permute_patch_x_p2g = np.vstack(self.h_permute_patch_list_p2g).T
            self.h_permute_patch_y_p2g = torch.arange(self.h_loc.shape[1])[None, :].repeat([self.h_loc.shape[0], 1])


        # priors
        self.p_loc = p_loc.detach().clone().to(device)
        self.p_log_scale = p_log_scale.detach().clone().to(device)
        self.h_p_loc = h_p_loc.detach().clone().to(device)
        self.h_p_log_scale = h_p_log_scale.detach().clone().to(device)
        self.hh_p_loc = hh_p_loc.detach().clone().to(device)
        self.hh_p_log_scale = hh_p_log_scale.detach().clone().to(device)

        # The compress progress is recorded here
        self.not_compress_mask_group = np.ones([number_of_datapoints, self.n_groups]).astype(bool)  # a mask indicating which group is NOT compressed (group_wise)
        self.compressed_groups_i = np.zeros([number_of_datapoints, self.n_groups])  # sample idx (final compressed result)
        self.compress_mask = torch.zeros_like(self.loc).to(device) # a mask indicating which part of the parameters is compressed (parameter-wise)
        self.group_sample = torch.zeros_like(self.loc).to(device) # samples for groups. will be updated if one group is compressed
        self.group_sample_std = 1e-15 + torch.zeros_like(self.loc).to(device) # std for samples. will be kept to zero always


        # The compress progress is recorded here
        self.h_not_compress_mask_group = np.ones([number_of_datapoints//16, self.h_n_groups]).astype(bool)  # a mask indicating which group is NOT compressed (group_wise)
        self.h_compressed_groups_i = np.zeros([number_of_datapoints//16, self.h_n_groups])  # sample idx (final compressed result)
        self.h_compress_mask = torch.zeros_like(self.h_loc).to(device) # a mask indicating which part of the parameters is compressed (parameter-wise)
        self.h_group_sample = torch.zeros_like(self.h_loc).to(device) # samples for groups. will be updated if one group is compressed
        self.h_group_sample_std = 1e-15 + torch.zeros_like(self.h_loc).to(device) # std for samples. will be kept to zero always

        self.hh_not_compress_mask_group = np.ones([number_of_datapoints//96, self.hh_n_groups]).astype(bool)  # a mask indicating which group is NOT compressed (group_wise)
        self.hh_compressed_groups_i = np.zeros([number_of_datapoints//96, self.hh_n_groups])  # sample idx (final compressed result)
        self.hh_compress_mask = torch.zeros_like(self.hh_loc).to(device) # a mask indicating which part of the parameters is compressed (parameter-wise)
        self.hh_group_sample = torch.zeros_like(self.hh_loc).to(device) # samples for groups. will be updated if one group is compressed
        self.hh_group_sample_std = 1e-15 + torch.zeros_like(self.hh_loc).to(device) # std for samples. will be kept to zero always


        self.g_samples = None # time samples for A* coding

        # the activation function
        self.act = Sine(w0)

        # calculate compression rate
        self.bpp = (self.n_groups * self.bit_per_group) / n_pixels + (self.h_n_groups * self.bit_per_group) / n_pixels / 96 * 6 + (self.hh_n_groups * self.bit_per_group) / n_pixels / 96
        print("Model Initialized. Expected bpp is %.2f" % self.bpp, flush=True)



        self.training = True


        # samples for different datapoints are the same
        # so they can be saved in dict to save time
        self.group_samples = {}
        self.h_group_samples = {}


    def group_to_layer(self, param, layer_idx):
        """
        Give an index of a layer, return the parameters (in a vector) corresponding to this layer
        """
        if layer_idx == 0:
            return param[..., :self.cum_param_sizes[layer_idx]]
        else:
            return param[..., self.cum_param_sizes[layer_idx-1]: self.cum_param_sizes[layer_idx]]

    def layer_to_weight(self, in_dim, out_dim, layer_param):
        """
        Give the parameters (in a vector) corresponding to this layer, return the weights and bias in matrix form
        """
        if layer_param.ndim == 2:
            bias = layer_param[:, :out_dim][:, None, :]
            weights = layer_param[:, out_dim:].reshape(-1, in_dim, out_dim)
            return weights, bias
        if layer_param.ndim == 3:
            bias = layer_param[:, :, :out_dim][:, :, None, :]
            weights = layer_param[:, :, out_dim:].reshape(layer_param.shape[0], layer_param.shape[1], in_dim, out_dim)
            return weights, bias

    def predict(self, x, random_seed=None, sample_size=1):
        """
        Prediction.
        """
        if random_seed != None:
            torch.manual_seed(random_seed)
        compress_mask = self.compress_mask
        group_sample = self.group_sample
        group_sample_std = self.group_sample_std

        if self.training:
            loc = self.loc * (1 - compress_mask) + group_sample * compress_mask
            scale = self.st(self.log_scale) * (1 - compress_mask) + group_sample_std * compress_mask
        else:
            loc = self.loc * (1 - compress_mask) + group_sample * compress_mask
            scale = self.st(self.log_scale) * (1 - compress_mask) + group_sample_std.round() * compress_mask

        if self.patch and self.mix_datapoints:
            # reorder columns back
            loc = loc[self.permute_patch_x_g2p, self.permute_patch_y_g2p]
            scale = scale[self.permute_patch_x_g2p, self.permute_patch_y_g2p]

        # reorder back to parameter order
        loc = loc[:, self.group_to_param]
        scale = scale[:, self.group_to_param]

        if self.patch == False:
            latent_code_loc = loc[:, self.cum_param_sizes[-1]:].reshape([-1, self.pixel_size//self.upsample_factor, self.pixel_size//self.upsample_factor, self.latent_dim])[None, ...]
            latent_code_scale = scale[:, self.cum_param_sizes[-1]:].reshape([-1, self.pixel_size//self.upsample_factor, self.pixel_size//self.upsample_factor, self.latent_dim])[None, ...].repeat([sample_size, 1, 1, 1, 1])
            latent_code = latent_code_loc + latent_code_scale * torch.randn_like(latent_code_scale) # n_samples, n_images, H, W, C
            latent_code = latent_code.permute([0, 1, 4, 2, 3])
            latent_code = latent_code.reshape([-1, latent_code.shape[2],  latent_code.shape[3], latent_code.shape[4]]) # n_samples*n_images, C, H, W
            coord_feature = self.coord_mapping(latent_code).permute([0, 2, 3, 1])
            coord_feature = coord_feature.reshape([sample_size, loc.shape[0], coord_feature.shape[1], coord_feature.shape[2], coord_feature.shape[3]])  # n_samples, n_images, H, W, C
            coord_feature = coord_feature.reshape([
                sample_size,
                loc.shape[0],
                -1,
                coord_feature.shape[-1]
            ]).permute([1, 0, 2, 3]) # n_images, n_samples, H * W, dim
        else:
            latent_code_loc = loc[:, self.cum_param_sizes[-1]:].reshape([-1, self.pixel_size // self.upsample_factor, self.pixel_size // self.upsample_factor, self.latent_dim])[None, ...]
            latent_code_scale = scale[:, self.cum_param_sizes[-1]:].reshape([-1, self.pixel_size // self.upsample_factor, self.pixel_size // self.upsample_factor, self.latent_dim])[None, ...].repeat([sample_size, 1, 1, 1, 1])
            latent_code = latent_code_loc + latent_code_scale * torch.randn_like(latent_code_scale)  # n_samples, n_images, H, W, C
            latent_code = latent_code.reshape(latent_code.shape[0], # n_samples
                                              512//64, # W
                                              768//64, # H
                                              latent_code.shape[-3],
                                              latent_code.shape[-2],
                                              latent_code.shape[-1]
                                              ).permute([
                0, 1, 3, 2, 4, 5
            ])
            latent_code = latent_code.reshape(latent_code.shape[0],
                                              512 // self.upsample_factor,
                                              768 // self.upsample_factor,
                                              latent_code.shape[-1]
                                              ).permute([
                0, 3, 1, 2
            ])
            coord_feature = self.coord_mapping(latent_code).permute([0, 2, 3, 1])
            coord_feature = coord_feature.reshape(sample_size,
                                                  512 // 64,
                                                  64,
                                                  768 // 64,
                                                  64,
                                                  coord_feature.shape[3]
                                                  ).permute([
                0, 1, 3, 2, 4, 5
            ])
            coord_feature = coord_feature.reshape(sample_size,
                                                  self.loc.shape[0],
                                                  64, 64, coord_feature.shape[-1]
                                                  ).reshape(sample_size,
                                                            self.loc.shape[0],
                                                            -1,
                                                            coord_feature.shape[-1]
                                                            ).permute([1, 0, 2, 3]) # n_images, n_samples, H * W, dim

        x = x[:, None, :, :].repeat([1, sample_size, 1, 1])  # n_images, 1, H * W, dim
        x = torch.cat([x, coord_feature], -1)


        loc = loc[:, :self.cum_param_sizes[-1]]
        scale = scale[:, :self.cum_param_sizes[-1]]

        # apply hyper prior
        loc = loc[:, None, :]
        scale = scale[:, None, :].repeat([1, sample_size, 1])
        sample_latent_all = loc + scale * torch.randn_like(scale)

        h_loc = self.h_loc * (1 - self.h_compress_mask) + self.h_group_sample * self.h_compress_mask
        h_scale = self.st(self.h_log_scale) * (1 - self.h_compress_mask) + self.h_group_sample_std * self.h_compress_mask
        if self.patch and self.mix_datapoints:
            # reorder columns back
            h_loc = h_loc[self.h_permute_patch_x_g2p, self.h_permute_patch_y_g2p]
            h_scale = h_scale[self.h_permute_patch_x_g2p, self.h_permute_patch_y_g2p]
        # reorder back to parameter order
        h_loc = h_loc[:, self.h_group_to_param]
        h_scale = h_scale[:, self.h_group_to_param] 

        h_loc = h_loc.reshape([-1, 6, h_loc.shape[-1]]).reshape([-1, 2, 3, h_loc.shape[-1]]) # n_images, 2, 3, dim
        h_loc = h_loc[:, :, None, :, None, :] # n_images, 2, 1, 3, 1, dim
        h_loc = h_loc.repeat([1, 1, 4, 1, 4, 1]) # n_images, 2, 4, 3, 4, dim
        h_loc = h_loc.reshape([-1, 96, h_loc.shape[-1]]).reshape([-1, h_loc.shape[-1]])

        h_scale = h_scale.reshape([-1, 6, h_scale.shape[-1]]).reshape([-1, 2, 3, h_scale.shape[-1]]) # n_images, 2, 3, dim
        h_scale = h_scale[:, :, None, :, None, :] # n_images, 2, 1, 3, 1, dim
        h_scale = h_scale.repeat([1, 1, 4, 1, 4, 1]) # n_images, 2, 4, 3, 4, dim
        h_scale = h_scale.reshape([-1, 96, h_scale.shape[-1]]).reshape([-1, h_scale.shape[-1]])

        h_loc = h_loc[:, None, :] # 1, sample_size, dim
        h_scale = h_scale[:, None, :].repeat([1, sample_size, 1])  # 1, sample_size, dim
            
        hyper_loc = h_loc + torch.randn_like(h_scale) * h_scale

       

        hh_loc = self.hh_loc * (1 - self.hh_compress_mask) + self.hh_group_sample * self.hh_compress_mask
        hh_scale = self.st(self.hh_log_scale) * (1 - self.hh_compress_mask) + self.hh_group_sample_std * self.hh_compress_mask
        hh_loc = hh_loc[:, self.hh_group_to_param]
        hh_scale = hh_scale[:, self.hh_group_to_param]



        hh_loc = hh_loc[:, None, :] # 1, sample_size, dim
        hh_scale = hh_scale[:, None, :].repeat([1, sample_size, 1])  # 1, sample_size, dim
        hh_param = hh_loc + hh_scale * torch.randn_like(hh_scale)
        hh_hyper_loc = hh_param

        
        sample_latent_all = sample_latent_all + hh_hyper_loc + hyper_loc


        for idx in range(self.n_layers):
            sample_latent = self.group_to_layer(sample_latent_all, idx)
            sample_latent = sample_latent @ self.param_mapping.A[idx]
            w, b = self.layer_to_weight(self.dims[idx], self.dims[idx + 1], sample_latent)
            x = (x @ w) + b # # N1, 1, N2, dim @ N1, sample_size, dim, dim'
            if idx != self.n_layers - 1:
                x = self.act(x)
        x = x[:, 0, :, :] if sample_size == 1 else x
        return x

    def calculate_kl_fast(self):

        p_scale = self.st(self.p_log_scale)
        kl_factor = self.kl_beta[:, self.group_idx].to(p_scale.device)
        kl = kl_divergence(Normal(self.loc, self.st(self.log_scale)), Normal(self.p_loc[None, :], p_scale[None, :]))
        assert kl.shape == kl_factor.shape
        kls = (kl * kl_factor).sum()

        h_p_scale = self.st(self.h_p_log_scale)
        h_kl_factor = self.h_kl_beta[:, self.h_group_idx].to(h_p_scale.device)
        h_kl = kl_divergence(Normal(self.h_loc, self.st(self.h_log_scale)), Normal(self.h_p_loc[None, :], h_p_scale[None, :]))
        assert h_kl.shape == h_kl_factor.shape
        h_kls = (h_kl * h_kl_factor).sum()

        hh_p_scale = self.st(self.hh_p_log_scale)
        hh_kl_factor = self.hh_kl_beta[:, self.hh_group_idx].to(hh_p_scale.device)
        hh_kl = kl_divergence(Normal(self.hh_loc, self.st(self.hh_log_scale)), Normal(self.hh_p_loc[None, :], hh_p_scale[None, :]))
        assert hh_kl.shape == hh_kl_factor.shape
        hh_kls = (hh_kl * hh_kl_factor).sum()

        return kls + h_kls + hh_kls

    def update_annealing_factors(self, update=True):
        # calculate KL first
        with torch.no_grad():
            p_scale = self.st(self.p_log_scale)
            kl = kl_divergence(Normal(self.loc, self.st(self.log_scale)),
                               Normal(self.p_loc[None, :], p_scale[None, :])).detach().cpu().numpy()
        kls = np.stack([np.bincount(self.group_idx, weights=kl[i]) for i in range(kl.shape[0])])

        with torch.no_grad():
            h_p_scale = self.st(self.h_p_log_scale)
            h_kl = kl_divergence(Normal(self.h_loc, self.st(self.h_log_scale)),
                                 Normal(self.h_p_loc[None, :], h_p_scale[None, :])).detach().cpu().numpy()
        h_kls = np.stack([np.bincount(self.h_group_idx, weights=h_kl[i]) for i in range(h_kl.shape[0])])


        with torch.no_grad():
            hh_p_scale = self.st(self.hh_p_log_scale)
            hh_kl = kl_divergence(Normal(self.hh_loc, self.st(self.hh_log_scale)),
                                 Normal(self.hh_p_loc[None, :], hh_p_scale[None, :])).detach().cpu().numpy()
        hh_kls = np.stack([np.bincount(self.hh_group_idx, weights=hh_kl[i]) for i in range(hh_kl.shape[0])])

        if update:
            new_kl_beta = self.kl_beta.clone()
            mask = (kls/np.log(2.) > (self.bit_per_group + self.kl_buffer - self.kl_boundary)).astype(float)
            new_kl_beta = new_kl_beta * torch.from_numpy(1 + self.eps_beta * mask).float()
            mask = (kls/np.log(2.) <= (self.bit_per_group - self.kl_buffer - self.kl_boundary)).astype(float)
            new_kl_beta = new_kl_beta / torch.from_numpy(1 + self.eps_beta * mask).float()
            new_kl_beta = torch.clamp(new_kl_beta, 0., 10000.)

            update_mask = self.not_compress_mask_group
            self.kl_beta = torch.where(torch.from_numpy(update_mask).bool(), new_kl_beta, self.kl_beta)

            h_new_kl_beta = self.h_kl_beta.clone()
            h_mask = (h_kls / np.log(2.) > (self.bit_per_group + self.kl_buffer - self.kl_boundary)).astype(float)
            h_new_kl_beta = h_new_kl_beta * torch.from_numpy(1 + self.eps_beta * h_mask).float()
            h_mask = (h_kls / np.log(2.) <= (self.bit_per_group - self.kl_buffer - self.kl_boundary)).astype(float)
            h_new_kl_beta = h_new_kl_beta / torch.from_numpy(1 + self.eps_beta * h_mask).float()
            h_new_kl_beta = torch.clamp(h_new_kl_beta, 0., 10000.)

            h_update_mask = self.h_not_compress_mask_group
            self.h_kl_beta = torch.where(torch.from_numpy(h_update_mask).bool(), h_new_kl_beta, self.h_kl_beta)

            hh_new_kl_beta = self.hh_kl_beta.clone()
            hh_mask = (hh_kls / np.log(2.) > (self.bit_per_group + self.kl_buffer - self.kl_boundary)).astype(float)
            hh_new_kl_beta = hh_new_kl_beta * torch.from_numpy(1 + self.eps_beta * hh_mask).float()
            hh_mask = (hh_kls / np.log(2.) <= (self.bit_per_group - self.kl_buffer - self.kl_boundary)).astype(float)
            hh_new_kl_beta = hh_new_kl_beta / torch.from_numpy(1 + self.eps_beta * hh_mask).float()
            hh_new_kl_beta = torch.clamp(hh_new_kl_beta, 0., 10000.)

            hh_update_mask = self.hh_not_compress_mask_group
            self.hh_kl_beta = torch.where(torch.from_numpy(hh_update_mask).bool(), hh_new_kl_beta, self.hh_kl_beta)


        return kls, h_kls, hh_kls

    def get_time_sample(self):
        group_sample_size = int(np.ceil(2 ** self.bit_per_group))
        np.random.seed(self.random_seed)
        log_u = np.log(np.random.rand(group_sample_size))
        g_loc = 0
        bound = g_loc - np.log(-log_u[0])
        g_samples = [bound]
        for i in range(1, group_sample_size):
            bound = -log_u[i] + np.exp(g_loc - bound)
            bound = g_loc - np.log(bound)
            g_samples.append(bound)
        g_samples = torch.from_numpy(np.array(g_samples))
        self.g_samples = g_samples

    def get_sample(self, group_idx, group_sample_size):
        try:
            return self.group_samples[(group_idx, group_sample_size)]
        except:
            # sample from prior
            sb = SobolEngine(-self.group_start_index[group_idx]+ self.group_end_index[group_idx], scramble=True, seed=self.random_seed)
            sb_sample = sb.draw(group_sample_size)
            samples = torch.from_numpy(norm.ppf(sb_sample))
            samples = torch.clamp(samples, -100, 100)
            self.group_samples[(group_idx, group_sample_size)] = samples.to(self.device)

            return self.group_samples[(group_idx, group_sample_size)]

    def h_get_sample(self, h_group_idx, group_sample_size):
        try:
            return self.h_group_samples[(h_group_idx, group_sample_size)]
        except:
            # sample from prior
            sb = SobolEngine(-self.h_group_start_index[h_group_idx]+ self.h_group_end_index[h_group_idx], scramble=True, seed=self.random_seed)
            sb_sample = sb.draw(group_sample_size)
            samples = torch.from_numpy(norm.ppf(sb_sample))
            samples = torch.clamp(samples, -100, 100)
            self.h_group_samples[(h_group_idx, group_sample_size)] = samples.to(self.device)

            return self.h_group_samples[(h_group_idx, group_sample_size)]

    def hh_get_sample(self, group_idx, group_sample_size):
        # sample from prior
        sb = SobolEngine(-self.hh_group_start_index[group_idx] + self.hh_group_end_index[group_idx], scramble=True, seed=self.random_seed)
        sb_sample = sb.draw(group_sample_size)
        samples = torch.from_numpy(norm.ppf(sb_sample))
        samples = torch.clamp(samples, -100, 100)
        return samples.to(self.device)

    def sample_group_image(self, image_idx, group_idx, group_sample_size):

        # A* Coding
        with torch.no_grad():
            samples = self.get_sample(group_idx, group_sample_size)

            # calculate the prior
            p_loc = self.p_loc[self.group_start_index[group_idx]: self.group_end_index[group_idx]]
            p_scale = self.st(self.p_log_scale[self.group_start_index[group_idx]: self.group_end_index[group_idx]])

            samples = p_loc + p_scale * samples
            log_p_samples = Normal(p_loc, p_scale).log_prob(samples).sum(-1)
            log_q_samples = Normal(self.loc[image_idx,
                                   self.group_start_index[group_idx]: self.group_end_index[group_idx]],
                                   self.st(self.log_scale[image_idx,
                                           self.group_start_index[group_idx]: self.group_end_index[group_idx]])
                                   ).log_prob(samples).sum(-1)  # (N1, n_samples)
            log_w = log_q_samples - log_p_samples

            # sample time (log space)
            if self.g_samples == None:
                self.get_time_sample()
            log_w = log_w + self.g_samples.to(log_w.device)[:group_sample_size]

            assert len(log_w.shape) == 1
            i = torch.argmax(log_w).item()
            z_i = samples[i, :].to(self.device)
        return i, z_i, log_w

    def h_sample_group_image(self, image_idx, group_idx, group_sample_size):

        # A* Coding
        with torch.no_grad():
            samples = self.h_get_sample(group_idx, group_sample_size)

            # calculate the prior
            p_loc = self.h_p_loc[self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]]
            p_scale = self.st(self.h_p_log_scale[self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]])

            samples = p_loc + p_scale * samples
            log_p_samples = Normal(p_loc, p_scale).log_prob(samples).sum(-1)
            log_q_samples = Normal(self.h_loc[image_idx,
                                   self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]],
                                   self.st(self.h_log_scale[image_idx,
                                           self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]])
                                   ).log_prob(samples).sum(-1)  # (N1, n_samples)
            log_w = log_q_samples - log_p_samples

            # sample time (log space)
            if self.g_samples == None:
                self.get_time_sample()
            log_w = log_w + self.g_samples.to(log_w.device)[:group_sample_size]

            assert len(log_w.shape) == 1
            i = torch.argmax(log_w).item()
            z_i = samples[i, :].to(self.device)
        return i, z_i, log_w


    def hh_sample_group_image(self, image_idx, group_idx, group_sample_size):

        # A* Coding
        with torch.no_grad():
            samples = self.hh_get_sample(group_idx, group_sample_size)

            # calculate the prior
            p_loc = self.hh_p_loc[self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]]
            p_scale = self.st(self.hh_p_log_scale[self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]])

            samples = p_loc + p_scale * samples
            log_p_samples = Normal(p_loc, p_scale).log_prob(samples).sum(-1)
            log_q_samples = Normal(self.hh_loc[image_idx,
                                   self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]],
                                   self.st(self.hh_log_scale[image_idx,
                                           self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]])
                                   ).log_prob(samples).sum(-1)  # (N1, n_samples)
            log_w = log_q_samples - log_p_samples

            # sample time (log space)
            if self.g_samples == None:
                self.get_time_sample()
            log_w = log_w + self.g_samples.to(log_w.device)[:group_sample_size]

            assert len(log_w.shape) == 1
            i = torch.argmax(log_w).item()
            z_i = samples[i, :].to(self.device)
        return i, z_i, log_w


    def compress_group_image(self, image_idx, group_idx):

        group_sample_size = int(np.ceil(2 ** self.bit_per_group))
        i, z_i, log_w = self.sample_group_image(image_idx, group_idx, group_sample_size)
        self.compressed_groups_i[image_idx, group_idx] = i
        self.not_compress_mask_group[image_idx, group_idx] = False
        self.group_sample[image_idx, self.group_start_index[group_idx]: self.group_end_index[group_idx]] = z_i.clone()
        self.compress_mask[image_idx, self.group_start_index[group_idx]: self.group_end_index[group_idx]] = 1
        self.kl_beta[image_idx, group_idx] = 0

        return i, z_i

    def h_compress_group_image(self, image_idx, group_idx):

        group_sample_size = int(np.ceil(2 ** self.bit_per_group))
        i, z_i, log_w = self.h_sample_group_image(image_idx, group_idx, group_sample_size)
        self.h_compressed_groups_i[image_idx, group_idx] = i
        self.h_not_compress_mask_group[image_idx, group_idx] = False
        self.h_group_sample[image_idx, self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]] = z_i.clone()
        self.h_compress_mask[image_idx, self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]] = 1
        self.h_kl_beta[image_idx, group_idx] = 0

        return i, z_i


    def hh_compress_group_image(self, image_idx, group_idx):

        group_sample_size = int(np.ceil(2 ** self.bit_per_group))
        i, z_i, log_w = self.hh_sample_group_image(image_idx, group_idx, group_sample_size)
        self.hh_compressed_groups_i[image_idx, group_idx] = i
        self.hh_not_compress_mask_group[image_idx, group_idx] = False
        self.hh_group_sample[image_idx, self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]] = z_i.clone()
        self.hh_compress_mask[image_idx, self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]] = 1
        self.hh_kl_beta[image_idx, group_idx] = 0

        return i, z_i

    def train(self,
              x, # (N1, N2, dim)
              y, # (N1, N2, dim)
              n_epochs,
              enforce_kl,
              optimizer,
              verbose,
              kl_adjust_epoch=None, # start to adjust after this epoch
              kl_adjust_gap=None, # asjust gap
              ):

        if kl_adjust_gap == None:
            kl_adjust_gap = self.kl_adjust_gap
        if kl_adjust_epoch == None:
            kl_adjust_epoch = self.kl_adjust_epoch

        for epoch in (tqdm(range(n_epochs)) if verbose else range(n_epochs)):
            sample_size = 5
            y_pred = self.predict(x=x, random_seed=epoch, sample_size=sample_size)  # explicitly enforce reproducing
            if sample_size != 1:
                loss = torch.mean((y_pred - y[:, None, :, :]) ** 2) * y.shape[0]
            else:
                loss = torch.mean((y_pred - y) ** 2) * y.shape[0]
            elbo = loss

            if enforce_kl:
                kl = self.calculate_kl_fast()
                elbo = elbo + kl
                if epoch >= kl_adjust_epoch and epoch % kl_adjust_gap == 0:
                    self.update_annealing_factors(update=True)
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

    def _compress_prepare(self,
                          image_paths,
                          feature_size,
                          ):
        X, Y = load_dataset(image_paths,
                            feature_size,
                            True,
                            self.patch_size)
        return X, Y

    def _compress_train(self,
                        x,
                        y,
                        n_epoch_kl,
                        verbose,
                        lr,
                        kl_adjust_epoch=None,  # start to adjust after this epoch
                        kl_adjust_gap=None,
                        ):

        if verbose:
            with torch.no_grad():
                y_pred = self.predict(x.to(self.device)).cpu()
                y_ori = y.cpu()
                if not self.patch:
                    psnr = batch_PSNR(y_ori.numpy(), y_pred.numpy()).mean()
                else:
                    psnr = PSNR(y_ori.numpy(), y_pred.numpy())
            print("Initialization... Average PSNR %.4f" % psnr, flush=True)
        kls, h_kls, hh_kls = self.update_annealing_factors(False)
        kl_bits = kls / np.log(2.)
        h_kl_bits = h_kls / np.log(2.)
        hh_kl_bits = hh_kls / np.log(2.)

        if verbose:
            print("Bits per group: ave %.2f" % kl_bits.mean() + " max %.2f" % kl_bits.max(), flush=True)
            print("Budget Success Ratio %.4f" % (kl_bits < self.bit_per_group).mean(), flush=True)

            print("", flush=True)

            print("Training...", flush=True)
        optimizer = Adam(self.parameters(), lr=lr)
        self.train(x=x,  # (N1, N2, dim)
                   y=y,  # (N1, N2, dim)
                   n_epochs=n_epoch_kl,
                   enforce_kl=True,
                   optimizer=optimizer,
                   verbose=verbose,
                   kl_adjust_epoch=kl_adjust_epoch,  # start to adjust after this epoch
                   kl_adjust_gap=kl_adjust_gap,  # adjust gap
                   )
        if verbose:
            with torch.no_grad():
                y_pred = self.predict(x.to(self.device)).cpu()
                y_ori = y.cpu()
                if not self.patch:
                    psnr = batch_PSNR(y_ori.numpy(), y_pred.numpy()).mean()
                else:
                    psnr = PSNR(y_ori.numpy(), y_pred.numpy())
            print("Training Finishes... PSNR %.4f" % psnr, flush=True)

        kls, h_kls, hh_kls = self.update_annealing_factors(False)
        kl_bits = kls / np.log(2.)
        h_kl_bits = h_kls / np.log(2.)
        hh_kl_bits = hh_kls / np.log(2.)

        if verbose:
            print("Bits per group: ave %.2f" % kl_bits.mean() + " max %.2f" % kl_bits.max(), flush=True)
            print("Budget Success Ratio %.4f" % (kl_bits < self.bit_per_group).mean(), flush=True)

        return psnr


    def _compress_compress(self,
                           x,
                           y,
                           n_epoch_compress,
                           verbose,
                           lr,
                           fine_tune_gap,
                           compress_from_largest=True,
                           ):
        if verbose:
            print("preparing samples...", flush=True)
        for i in range(self.n_groups):
            self.get_sample(i, 2**16)
        if verbose:
            print("Compressing...", flush=True)

        try:
            test = self.compressed_num
        except:
            self.compressed_num = 0
            self.print_record = np.zeros(10)
            self.print_cursor = 0
        try:
            test = self.h_compressed_num
        except:
            self.h_compressed_num = 0
            self.h_print_record = np.zeros(10)
            self.h_print_cursor = 0
        try:
            test = self.hh_compressed_num
        except:
            self.hh_compressed_num = 0
            self.hh_print_record = np.zeros(10)
            self.hh_print_cursor = 0
        epo = n_epoch_compress


        # group_idx = -1
        for _i in tqdm(range(self.hh_compressed_num, self.hh_n_groups)):
            for image_idx in range(self.hh_loc.shape[0]):
                group_idx = _i
                if compress_from_largest:
                    # find the largest KL idx
                    kl_bits, h_kl_bits, hh_kl_bits = self.update_annealing_factors(False)
                    hh_kl_bits = hh_kl_bits[image_idx] / np.log(2.)
                    hh_mask = (self.hh_not_compress_mask_group[image_idx] == False)
                    hh_kl_bits[hh_mask] = -1e10
                    group_idx = hh_kl_bits.argmax()
                self.hh_compress_group_image(image_idx, group_idx)

            self.hh_compressed_num += 1

            if self.hh_compressed_num % fine_tune_gap == 0:
                optimizer = Adam(self.parameters(), lr=lr) # reinitialize the momentums
                self.train(x, # (N1, N2, dim)
                           y, # (N1, N2, dim)
                           n_epochs=100,#epo,
                           enforce_kl=True,
                           optimizer=optimizer,
                           verbose=False,
                           kl_adjust_epoch=0,
                           kl_adjust_gap=10,
                           )
            if 10 * (self.hh_compressed_num / self.hh_n_groups) > self.hh_print_cursor:
                if self.hh_print_record[self.hh_print_cursor] == 0:
                    if verbose:
                        with torch.no_grad():
                            y_pred = self.predict(x.to(self.device)).cpu()
                            y_ori = y.cpu()
                            if not self.patch:
                                psnr = batch_PSNR(y_ori.numpy(), y_pred.numpy()).mean()
                            else:
                                psnr = PSNR(y_ori.numpy(), y_pred.numpy())
                        try:
                            kl_bits, h_kl_bits, hh_kl_bits  = self.update_annealing_factors(False)
                            hh_kl_bits /= np.log(2.)
                            mask = (self.hh_not_compress_mask_group == True)
                            kl_bits = hh_kl_bits.flatten()[mask.flatten()]
                            print("Compress progress: %d" % (100 * self.hh_compressed_num / self.hh_n_groups) + " Average PSNR %.4f" % psnr, "Avg KL/bit %.4f"%kl_bits.mean(), "Max KL/bit %.4f"%kl_bits.max(), flush=True)
                        except:
                            pass
                    self.hh_print_record[self.hh_print_cursor] = 1
                    self.hh_print_cursor += 1
        print(' ')

        # group_idx = -1
        for _i in tqdm(range(self.h_compressed_num, self.h_n_groups)):
            for image_idx in range(self.h_loc.shape[0]):
                group_idx = _i
                if compress_from_largest:
                    # find the largest KL idx
                    kl_bits, h_kl_bits, hh_kl_bits = self.update_annealing_factors(False)
                    h_kl_bits = h_kl_bits[image_idx] / np.log(2.)
                    h_mask = (self.h_not_compress_mask_group[image_idx] == False)
                    h_kl_bits[h_mask] = -1e10
                    group_idx = h_kl_bits.argmax()
                self.h_compress_group_image(image_idx, group_idx)

            self.h_compressed_num += 1

            if self.h_compressed_num % fine_tune_gap == 0:
                optimizer = Adam(self.parameters(), lr=lr) # reinitialize the momentums
                self.train(x, # (N1, N2, dim)
                           y, # (N1, N2, dim)
                           n_epochs=100,#epo,
                           enforce_kl=True,
                           optimizer=optimizer,
                           verbose=False,
                           kl_adjust_epoch=0,
                           kl_adjust_gap=10,
                           )
            if 10 * (self.h_compressed_num / self.h_n_groups) > self.h_print_cursor:
                if self.h_print_record[self.h_print_cursor] == 0:
                    if verbose:
                        with torch.no_grad():
                            y_pred = self.predict(x.to(self.device)).cpu()
                            y_ori = y.cpu()
                            if not self.patch:
                                psnr = batch_PSNR(y_ori.numpy(), y_pred.numpy()).mean()
                            else:
                                psnr = PSNR(y_ori.numpy(), y_pred.numpy())
                        try:
                            kl_bits, h_kl_bits, hh_kl_bits  = self.update_annealing_factors(False)
                            h_kl_bits /= np.log(2.)
                            mask = (self.h_not_compress_mask_group == True)
                            kl_bits = h_kl_bits.flatten()[mask.flatten()]
                            print("Compress progress: %d" % (100 * self.h_compressed_num / self.h_n_groups) + " Average PSNR %.4f" % psnr, "Avg KL/bit %.4f"%kl_bits.mean(), "Max KL/bit %.4f"%kl_bits.max(), flush=True)
                        except:
                            pass
                    self.h_print_record[self.h_print_cursor] = 1
                    self.h_print_cursor += 1
        print(' ')

        # group_idx = -1
        for _i in tqdm(range(self.compressed_num, self.n_groups)):
            for image_idx in range(self.loc.shape[0]):
                group_idx = _i
                if compress_from_largest:
                    # find the largest KL idx
                    kl_bits, h_kl_bits, hh_kl_bits = self.update_annealing_factors(False)
                    kl_bits = kl_bits[image_idx] / np.log(2.)
                    mask = (self.not_compress_mask_group[image_idx] == False)
                    kl_bits[mask] = -1e10
                    group_idx = kl_bits.argmax()
                self.compress_group_image(image_idx, group_idx)

            self.compressed_num += 1

            if self.compressed_num % fine_tune_gap == 0:
                optimizer = Adam(self.parameters(), lr=lr) # reinitialize the momentums
                self.train(x, # (N1, N2, dim)
                           y, # (N1, N2, dim)
                           n_epochs=epo,
                           enforce_kl=True,
                           optimizer=optimizer,
                           verbose=False,
                           kl_adjust_epoch=0,
                           kl_adjust_gap=10,
                           )
            if 10 * (self.compressed_num / self.n_groups) > self.print_cursor:
                if self.print_record[self.print_cursor] == 0:
                    if verbose:
                        with torch.no_grad():
                            y_pred = self.predict(x.to(self.device)).cpu()
                            y_ori = y.cpu()
                            if not self.patch:
                                psnr = batch_PSNR(y_ori.numpy(), y_pred.numpy()).mean()
                            else:
                                psnr = PSNR(y_ori.numpy(), y_pred.numpy())
                        try:
                            kl_bits, h_kl_bits, hh_kl_bits = self.update_annealing_factors(False)
                            kl_bits /= np.log(2.)
                            mask = (self.not_compress_mask_group == True)
                            kl_bits = kl_bits.flatten()[mask.flatten()]
                            print("Compress progress: %d" % (100 * self.compressed_num / self.n_groups) + " Average PSNR %.4f" % psnr, "Avg KL/bit %.4f"%kl_bits.mean(), "Max KL/bit %.4f"%kl_bits.max(), flush=True)
                        except:
                            pass
                    self.print_record[self.print_cursor] = 1
                    self.print_cursor += 1

        with torch.no_grad():
            y_pred = self.predict(x.to(self.device)).cpu()
            y_ori = y.cpu()
            if not self.patch:
                psnr = batch_PSNR(y_ori.numpy(), y_pred.numpy()).mean()
            else:
                psnr = PSNR(y_ori.numpy(), y_pred.numpy())
        if verbose:
            print("Compression is Finished... PSNR %.4f" % psnr, flush=True)
        return psnr














