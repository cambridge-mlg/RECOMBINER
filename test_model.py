import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import kl_divergence, Normal
from torch.quasirandom import SobolEngine
from scipy.stats import norm

from utils import PSNR, batch_PSNR, batch_RMSD, count_net_params

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
                 # network architectures and dataset info
                 in_dim,
                 hidden_dims,
                 out_dim,
                 number_of_datapoints,
                 upsample_factors,
                 latent_dim,
                 pixel_sizes,
                 dataset,
                 patch,
                 hierarchical_patch_nums,

                 # learned mappings and priors
                 linear_transform=None,
                 upsample_net=None,

                 p_loc=None,
                 p_log_scale=None,
                 init_log_scale=-4.,
                 param_to_group=None,
                 group_to_param=None,
                 n_groups=None,
                 group_start_index=None,
                 group_end_index=None,
                 group_idx=None, 

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

                 # other hyperparameters
                 w0=30.,
                 c=6.,
                 random_seed=42,
                 device='cuda',
                 kl_upper_buffer=0.,
                 kl_lower_buffer=0.4,
                 kl_adjust_gap=10,
                 initial_beta=1e-8,
                 beta_step_size=0.05,
                 ):
        """
        RECOMBINER model for compression.
        """
        super().__init__()
        self.bit_per_group = 16 # bits assigned to each group (i.e., block)
        self.n_layers = len(hidden_dims) + 1
        self.dims = [in_dim] + hidden_dims + [out_dim]
        self.st = lambda x: F.softplus(x, beta=1, threshold=20) / 6 # scale transform function 
        self.upsample_factors = upsample_factors
        self.latent_dim = latent_dim
        self.patch = patch
        self.pixel_sizes = pixel_sizes
        self.linear_transform = linear_transform
        self.upsample_net = upsample_net

        self.device = device
        self.dataset = dataset
        self.random_seed = random_seed

        # fix the parameters in self.upsample_net and self.linear_transform
        try:
            for param in self.linear_transform.parameters():
                param.requires_grad = False
        except:
            pass
        try:
            for param in self.upsample_net.parameters():
                param.requires_grad = False
        except:
            pass

        # calculate number of parameters in INR
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

        # parameter and its grouping in the second level
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