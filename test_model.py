import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import kl_divergence, Normal
from torch.quasirandom import SobolEngine
from scipy.stats import norm

from utils import map_lpe_to_inr_inputs, \
                  map_hierarchical_model_to_int_weights, \
                  count_layer_params, \
                  count_net_params, \
                  PSNR, \
                  batch_PSNR, \
                  metric

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
                 data_dim,
                 pixel_sizes,
                 patch,
                 patch_nums,
                 hierarchical_patch_nums,
                 dataset,

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
        self.data_dim = data_dim
        self.patch = patch
        self.patch_nums = patch_nums
        self.pixel_sizes = pixel_sizes
        self.linear_transform = linear_transform
        self.upsample_net = upsample_net
        self.hierarchical_patch_nums = hierarchical_patch_nums

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

        # parameter and its grouping (latent positional encodings and latent INR weights in the lowest level)
        self.param_to_group = param_to_group
        self.group_to_param = group_to_param
        self.n_groups = n_groups
        self.group_start_index = group_start_index
        self.group_end_index = group_end_index
        self.group_idx = group_idx
        number_of_params = p_loc.shape[0]
        loc_data = p_loc[None, :].repeat([number_of_datapoints, 1]).to(device)
        self.log_scale = nn.Parameter(torch.zeros([number_of_datapoints, number_of_params]) + init_log_scale)
        self.loc = nn.Parameter(loc_data.clone())

        # parameter and its grouping in the second level
        if self.patch:
            self.h_param_to_group = h_param_to_group
            self.h_group_to_param = h_group_to_param
            self.h_n_groups = h_n_groups
            self.h_group_start_index = h_group_start_index
            self.h_group_end_index = h_group_end_index
            self.h_group_idx = h_group_idx
            h_number_of_params = h_p_loc.shape[0]
            h_loc_data = h_p_loc[None, :].repeat([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level2']), 1]).to(device)
            self.h_log_scale = nn.Parameter(torch.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level2']), h_number_of_params]) + h_init_log_scale)
            self.h_loc = nn.Parameter(h_loc_data.clone())

            # parameter and its grouping in the third level
            self.hh_param_to_group = hh_param_to_group
            self.hh_group_to_param = hh_group_to_param
            self.hh_n_groups = hh_n_groups
            self.hh_group_start_index = hh_group_start_index
            self.hh_group_end_index = hh_group_end_index
            self.hh_group_idx = hh_group_idx
            hh_number_of_params = hh_p_loc.shape[0]
            hh_loc_data = hh_p_loc[None, :].repeat([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level3']), 1]).to(device)
            self.hh_log_scale = nn.Parameter(torch.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level3']), hh_number_of_params]) + hh_init_log_scale)
            self.hh_loc = nn.Parameter(hh_loc_data.clone())

        # KL scale factor of each group
        self.kl_beta = torch.zeros([number_of_datapoints, self.n_groups]) + initial_beta
        if self.patch:
            self.h_kl_beta = torch.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level2']), self.h_n_groups]) + initial_beta
            self.hh_kl_beta = torch.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level3']), self.hh_n_groups]) + initial_beta

        self.beta_step_size = beta_step_size
        self.kl_upper_buffer = kl_upper_buffer
        self.kl_lower_buffer = kl_lower_buffer
        self.kl_adjust_gap = kl_adjust_gap

        # if patching, randomly permute the columns to allocate budgets better
        # here we store self.permute_patch_x_g2p, self.permute_patch_y_g2p, self.permute_patch_x_p2g, self.permute_patch_y_p2g
        # and when reconstructing the signal, we first permute the saved tensor by [...x_g2p, ...y_g2p], indicating the order are mapped from group order to parameter order
        if self.patch:
            self.permute_patch_list_g2p = []  # permute each dimension of loc/scale (colmun)
            self.permute_patch_list_p2g = []
            for dim_idx in range(self.loc.shape[1]):
                np.random.seed(dim_idx)
                patch_order = np.random.choice(self.loc.shape[0], self.loc.shape[0], False)
                self.permute_patch_list_g2p.append(patch_order)
                self.permute_patch_list_p2g.append(np.argsort(patch_order))
                np.random.seed(None)
            self.permute_patch_x_g2p = np.vstack(self.permute_patch_list_g2p).T
            self.permute_patch_y_g2p = torch.arange(self.loc.shape[1])[None, :].repeat([self.loc.shape[0], 1])
            self.permute_patch_x_p2g = np.vstack(self.permute_patch_list_p2g).T
            self.permute_patch_y_p2g = torch.arange(self.loc.shape[1])[None, :].repeat([self.loc.shape[0], 1])
            
            # this permutation is also applied on the 2nd level
            self.h_permute_patch_list_g2p = [] 
            self.h_permute_patch_list_p2g = []
            for dim_idx in range(self.h_loc.shape[1]):
                np.random.seed(dim_idx)
                patch_order = np.random.choice(self.h_loc.shape[0], self.h_loc.shape[0], False)
                self.h_permute_patch_list_g2p.append(patch_order)
                self.h_permute_patch_list_p2g.append(np.argsort(patch_order))
                np.random.seed(None)
            self.h_permute_patch_x_g2p = np.vstack(self.h_permute_patch_list_g2p).T
            self.h_permute_patch_y_g2p = torch.arange(self.h_loc.shape[1])[None, :].repeat([self.h_loc.shape[0], 1])
            self.h_permute_patch_x_p2g = np.vstack(self.h_permute_patch_list_p2g).T
            self.h_permute_patch_y_p2g = torch.arange(self.h_loc.shape[1])[None, :].repeat([self.h_loc.shape[0], 1])

        # priors
        self.p_loc = p_loc.detach().clone().to(device)
        self.p_log_scale = p_log_scale.detach().clone().to(device)
        if self.patch:
            self.h_p_loc = h_p_loc.detach().clone().to(device)
            self.h_p_log_scale = h_p_log_scale.detach().clone().to(device)
            self.hh_p_loc = hh_p_loc.detach().clone().to(device)
            self.hh_p_log_scale = hh_p_log_scale.detach().clone().to(device)

        # The compress progress is recorded here
        self.compressed_mask_groupwise = np.zeros([number_of_datapoints, self.n_groups]).astype(bool)  # a mask indicating which group is compressed (group_wise)
        self.compressed_idx_groupwise = np.zeros([number_of_datapoints, self.n_groups])  # sample idx (final compressed result)
        self.compressed_mask = torch.zeros_like(self.loc).to(device) # a mask indicating which part of the parameters is compressed (parameter-wise)
        self.compressed_sample = torch.zeros_like(self.loc).to(device) # samples for groups. will be updated if one group is compressed
        self.compressed_sample_std = 1e-15 + torch.zeros_like(self.loc).to(device) # std for samples. will be kept to zero always

        if self.patch:
            self.h_compressed_mask_groupwise = np.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level2']), self.h_n_groups]).astype(bool)  # a mask indicating which group is compressed (group_wise)
            self.h_compressed_idx_groupwise = np.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level2']), self.h_n_groups])  # sample idx (final compressed result)
            self.h_compressed_mask = torch.zeros_like(self.h_loc).to(device) # a mask indicating which part of the parameters is compressed (parameter-wise)
            self.h_compressed_sample = torch.zeros_like(self.h_loc).to(device) # samples for groups. will be updated if one group is compressed
            self.h_compressed_sample_std = 1e-15 + torch.zeros_like(self.h_loc).to(device) # std for samples. will be kept to zero always

            self.hh_compressed_mask_groupwise = np.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level3']), self.hh_n_groups]).astype(bool)  # a mask indicating which group is compressed (group_wise)
            self.hh_compressed_idx_groupwise = np.zeros([number_of_datapoints//np.prod(self.hierarchical_patch_nums['level3']), self.hh_n_groups])  # sample idx (final compressed result)
            self.hh_compressed_mask = torch.zeros_like(self.hh_loc).to(device) # a mask indicating which part of the parameters is compressed (parameter-wise)
            self.hh_compressed_sample = torch.zeros_like(self.hh_loc).to(device) # samples for groups. will be updated if one group is compressed
            self.hh_compressed_sample_std = 1e-15 + torch.zeros_like(self.hh_loc).to(device) # std for samples. will be kept to zero always

        self.g_samples = None # gumbel samples for A* coding

        # the activation function
        self.act = Sine(w0)

        # calculate compression rate
        if self.patch:
            self.bpp = (self.n_groups * self.bit_per_group) / np.prod(pixel_sizes) + (self.h_n_groups * self.bit_per_group) / np.prod(pixel_sizes) / np.prod(self.hierarchical_patch_nums['level2']) + (self.hh_n_groups * self.bit_per_group) / np.prod(pixel_sizes) / np.prod(self.hierarchical_patch_nums['level3'])
        else:
            self.bpp = (self.n_groups * self.bit_per_group) / np.prod(pixel_sizes)
        if self.dataset == 'audio':
            self.bpp =  self.bpp / (3/48000) / 1000 
        print("Model Initialized. Expected bpp is %.2f" % self.bpp, flush=True)

        # prior samples for different data points are the same
        # so they can be saved in dict to save time
        self.group_samples = {}
        if self.patch:
            self.h_group_samples = {}
            self.hh_group_samples = {}

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
        if random_seed != None:
            torch.manual_seed(random_seed)
        compressed_mask = self.compressed_mask
        compressed_sample = self.compressed_sample
        compressed_sample_std = self.compressed_sample_std
        loc = self.loc * (1 - compressed_mask) + compressed_sample * compressed_mask
        scale = self.st(self.log_scale) * (1 - compressed_mask) + compressed_sample_std * compressed_mask

        if self.patch:
            # permute columns back
            loc = loc[self.permute_patch_x_g2p, self.permute_patch_y_g2p]
            scale = scale[self.permute_patch_x_g2p, self.permute_patch_y_g2p]
        # permute back to parameter order
        loc = loc[:, self.group_to_param]
        scale = scale[:, self.group_to_param]

        # get latent positional encodings' parameters and map to input of INR
        lpe_loc = loc[None, :, self.cum_param_sizes[-1]:]
        lpe_scale = scale[None, :, self.cum_param_sizes[-1]:].repeat([sample_size, 1, 1])
        lpe = lpe_loc + lpe_scale * torch.randn_like(lpe_scale)
        pe = map_lpe_to_inr_inputs(self.upsample_net,
                                   lpe,
                                   self.latent_dim,
                                   self.pixel_sizes,
                                   self.upsample_factors,
                                   self.patch,
                                   self.patch_nums,
                                   self.data_dim) # pe shape (data_num, sample_size, total_pixel_num, pe_dim)
        x = x[:, None, :, :].repeat([1, sample_size, 1, 1]) 
        x = torch.cat([x, pe], -1)

        # get latent network parameters
        loc = loc[:, :self.cum_param_sizes[-1]]
        scale = scale[:, :self.cum_param_sizes[-1]]

        if self.patch:
            h_loc = self.h_loc * (1 - self.h_compressed_mask) + self.h_compressed_sample * self.h_compressed_mask
            h_scale = self.st(self.h_log_scale) * (1 - self.h_compressed_mask) + self.h_compressed_sample_std * self.h_compressed_mask
            h_loc = h_loc[self.h_permute_patch_x_g2p, self.h_permute_patch_y_g2p]
            h_scale = h_scale[self.h_permute_patch_x_g2p, self.h_permute_patch_y_g2p]
            h_loc = h_loc[:, self.h_group_to_param]
            h_scale = h_scale[:, self.h_group_to_param] 

            hh_loc = self.hh_loc * (1 - self.hh_compressed_mask) + self.hh_compressed_sample * self.hh_compressed_mask
            hh_scale = self.st(self.hh_log_scale) * (1 - self.hh_compressed_mask) + self.hh_compressed_sample_std * self.hh_compressed_mask
            hh_loc = hh_loc[:, self.hh_group_to_param]
            hh_scale = hh_scale[:, self.hh_group_to_param]
        else:
            h_loc = None
            h_scale = None
            hh_loc = None
            hh_scale = None

        h_w = map_hierarchical_model_to_int_weights(use_hierarchical_model=self.patch,
                                                    loc=loc, scale=scale,
                                                    h_loc=h_loc, h_scale=h_scale,
                                                    hh_loc=hh_loc, hh_scale=hh_scale,
                                                    sample_size=sample_size,
                                                    hierarchical_patch_nums=self.hierarchical_patch_nums,
                                                    patch_nums=self.patch_nums,
                                                    data_dim=self.data_dim)


        for idx in range(self.n_layers):
            _h_w = self.group_to_layer(h_w, idx)
            _h_w = _h_w @ self.linear_transform.A[idx]
            w, b = self.layer_to_weight(self.dims[idx], self.dims[idx + 1], _h_w)
            x = (x @ w) + b # # N1, 1, N2, dim @ N1, sample_size, dim, dim'
            if idx != self.n_layers - 1:
                x = self.act(x)
        x = x[:, 0, :, :] if sample_size == 1 else x
        return x
    
    def calculate_kl(self):
        p_scale = self.st(self.p_log_scale)
        kl_factor = self.kl_beta[:, self.group_idx].to(p_scale.device)
        kl = kl_divergence(Normal(self.loc, self.st(self.log_scale)), Normal(self.p_loc[None, :], p_scale[None, :]))
        assert kl.shape == kl_factor.shape
        kls = (kl * kl_factor).sum()

        if self.patch:
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

        return (kls + h_kls + hh_kls) if self.patch else kls
    
    def update_annealing_factors(self, update=True):
        """
        first calculate KL per group; and update kl_beta according to KL
        """
        # calculate KL first
        with torch.no_grad():
            p_scale = self.st(self.p_log_scale)
            kl = kl_divergence(Normal(self.loc, self.st(self.log_scale)),
                               Normal(self.p_loc[None, :], p_scale[None, :])).detach().cpu().numpy()
        kls = np.stack([np.bincount(self.group_idx, weights=kl[i]) for i in range(kl.shape[0])])

        if self.patch:
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
            mask = (kls/np.log(2.) > (self.bit_per_group + self.kl_upper_buffer)).astype(float)
            new_kl_beta = new_kl_beta * torch.from_numpy(1 + self.beta_step_size * mask).float()
            mask = (kls/np.log(2.) <= (self.bit_per_group - self.kl_lower_buffer)).astype(float)
            new_kl_beta = new_kl_beta / torch.from_numpy(1 + self.beta_step_size * mask).float()
            new_kl_beta = torch.clamp(new_kl_beta, 0., 10000.)

            update_mask = 1 - self.compressed_mask_groupwise # only update the beta for not yet compressed groups
            self.kl_beta = torch.where(torch.from_numpy(update_mask).bool(), new_kl_beta, self.kl_beta)
            
            if self.patch:
                h_new_kl_beta = self.h_kl_beta.clone()
                h_mask = (h_kls / np.log(2.) > (self.bit_per_group + self.kl_upper_buffer)).astype(float)
                h_new_kl_beta = h_new_kl_beta * torch.from_numpy(1 + self.beta_step_size * h_mask).float()
                h_mask = (h_kls / np.log(2.) <= (self.bit_per_group - self.kl_lower_buffer)).astype(float)
                h_new_kl_beta = h_new_kl_beta / torch.from_numpy(1 + self.beta_step_size * h_mask).float()
                h_new_kl_beta = torch.clamp(h_new_kl_beta, 0., 10000.)

                h_update_mask = 1 - self.h_compressed_mask_groupwise
                self.h_kl_beta = torch.where(torch.from_numpy(h_update_mask).bool(), h_new_kl_beta, self.h_kl_beta)

                hh_new_kl_beta = self.hh_kl_beta.clone()
                hh_mask = (hh_kls / np.log(2.) > (self.bit_per_group + self.kl_upper_buffer)).astype(float)
                hh_new_kl_beta = hh_new_kl_beta * torch.from_numpy(1 + self.beta_step_size * hh_mask).float()
                hh_mask = (hh_kls / np.log(2.) <= (self.bit_per_group - self.kl_lower_buffer)).astype(float)
                hh_new_kl_beta = hh_new_kl_beta / torch.from_numpy(1 + self.beta_step_size * hh_mask).float()
                hh_new_kl_beta = torch.clamp(hh_new_kl_beta, 0., 10000.)

                hh_update_mask = 1 - self.hh_compressed_mask_groupwise
                self.hh_kl_beta = torch.where(torch.from_numpy(hh_update_mask).bool(), hh_new_kl_beta, self.hh_kl_beta)

        if self.patch:
            return kls, h_kls, hh_kls
        else:
            return kls
        
    def get_gumbel_sample(self):
        """
        Generate gumbel samples and save for reuse.
        All groups will share the same gumbel samples to save time.
        """
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
        """
        generate samples from the prior for group_idx and same them in self.group_samples.
        """
        try:
            return self.group_samples[(group_idx, group_sample_size)]
        except:
            # sample from prior
            param_size = -self.group_start_index[group_idx] + self.group_end_index[group_idx]
            samples = self.get_sobol_normal_sample(param_size, group_sample_size)
            self.group_samples[(group_idx, group_sample_size)] = samples.to(self.device)

            return self.group_samples[(group_idx, group_sample_size)]

    def h_get_sample(self, h_group_idx, group_sample_size):
        try:
            return self.h_group_samples[(h_group_idx, group_sample_size)]
        except:
            # sample from prior
            param_size = -self.h_group_start_index[h_group_idx] + self.h_group_end_index[h_group_idx]
            samples = self.get_sobol_normal_sample(param_size, group_sample_size)
            self.h_group_samples[(h_group_idx, group_sample_size)] = samples.to(self.device)
            return self.h_group_samples[(h_group_idx, group_sample_size)]

    def hh_get_sample(self, hh_group_idx, group_sample_size):
        try:
            return self.hh_group_samples[(hh_group_idx, group_sample_size)]
        except:
            # sample from prior
            param_size = -self.hh_group_start_index[hh_group_idx] + self.hh_group_end_index[hh_group_idx]
            samples = self.get_sobol_normal_sample(param_size, group_sample_size)
            self.hh_group_samples[(hh_group_idx, group_sample_size)] = samples.to(self.device)
            return samples.to(self.device)
    
    def get_sobol_normal_sample(self, param_size, sample_size):
        sb = SobolEngine(param_size, scramble=True, seed=self.random_seed)
        sb_sample = sb.draw(sample_size)
        samples = torch.from_numpy(norm.ppf(sb_sample))
        samples = torch.clamp(samples, -100, 100)
        return samples


    def sample_group(self, row_idx, group_idx, group_sample_size):
        """
        Encode the ```group_idx```-th group at ```row_idx``` by A* coding.

        """

        # A* Coding
        with torch.no_grad():
            samples = self.get_sample(group_idx, group_sample_size)

            # calculate the prior
            p_loc = self.p_loc[self.group_start_index[group_idx]: self.group_end_index[group_idx]]
            p_scale = self.st(self.p_log_scale[self.group_start_index[group_idx]: self.group_end_index[group_idx]])
            samples = p_loc + p_scale * samples

            # calculate the likelihoods
            log_p_samples = Normal(p_loc, p_scale).log_prob(samples).sum(-1)
            log_q_samples = Normal(self.loc[row_idx, self.group_start_index[group_idx]: self.group_end_index[group_idx]],
                                   self.st(self.log_scale[row_idx, self.group_start_index[group_idx]: self.group_end_index[group_idx]])
                                   ).log_prob(samples).sum(-1)  # (N1, n_samples)
            log_w = log_q_samples - log_p_samples

            # sample gumbel noise
            if self.g_samples == None:
                self.get_gumbel_sample()
            log_w = log_w + self.g_samples.to(log_w.device)[:group_sample_size]

            # encode the sample
            assert len(log_w.shape) == 1
            i = torch.argmax(log_w).item()
            z_i = samples[i, :].to(self.device)

        return i, z_i, log_w

    def h_sample_group(self, row_idx, group_idx, group_sample_size):

        # A* Coding
        with torch.no_grad():
            samples = self.h_get_sample(group_idx, group_sample_size)

            p_loc = self.h_p_loc[self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]]
            p_scale = self.st(self.h_p_log_scale[self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]])
            samples = p_loc + p_scale * samples

            log_p_samples = Normal(p_loc, p_scale).log_prob(samples).sum(-1)
            log_q_samples = Normal(self.h_loc[row_idx, self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]],
                                   self.st(self.h_log_scale[row_idx, self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]])
                                   ).log_prob(samples).sum(-1)  # (N1, n_samples)
            log_w = log_q_samples - log_p_samples

            if self.g_samples == None:
                self.get_gumbel_sample()
            log_w = log_w + self.g_samples.to(log_w.device)[:group_sample_size]

            assert len(log_w.shape) == 1
            i = torch.argmax(log_w).item()
            z_i = samples[i, :].to(self.device)
        return i, z_i, log_w


    def hh_sample_group(self, row_idx, group_idx, group_sample_size):

        # A* Coding
        with torch.no_grad():
            samples = self.hh_get_sample(group_idx, group_sample_size)

            p_loc = self.hh_p_loc[self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]]
            p_scale = self.st(self.hh_p_log_scale[self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]])
            samples = p_loc + p_scale * samples

            log_p_samples = Normal(p_loc, p_scale).log_prob(samples).sum(-1)
            log_q_samples = Normal(self.hh_loc[row_idx, self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]],
                                   self.st(self.hh_log_scale[row_idx, self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]])
                                   ).log_prob(samples).sum(-1)  # (N1, n_samples)
            log_w = log_q_samples - log_p_samples

            if self.g_samples == None:
                self.get_gumbel_sample()
            log_w = log_w + self.g_samples.to(log_w.device)[:group_sample_size]

            assert len(log_w.shape) == 1
            i = torch.argmax(log_w).item()
            z_i = samples[i, :].to(self.device)
        return i, z_i, log_w
    
    def compress_group(self, row_idx, group_idx):
        group_sample_size = int(np.ceil(2 ** self.bit_per_group))
        i, z_i, log_w = self.sample_group(row_idx, group_idx, group_sample_size)
        self.compressed_idx_groupwise[row_idx, group_idx] = i
        self.compressed_mask_groupwise[row_idx, group_idx] = True
        self.compressed_sample[row_idx, self.group_start_index[group_idx]: self.group_end_index[group_idx]] = z_i.clone()
        self.compressed_mask[row_idx, self.group_start_index[group_idx]: self.group_end_index[group_idx]] = 1
        self.kl_beta[row_idx, group_idx] = 0 

        return i, z_i

    def h_compress_group(self, row_idx, group_idx):

        group_sample_size = int(np.ceil(2 ** self.bit_per_group))
        i, z_i, log_w = self.h_sample_group(row_idx, group_idx, group_sample_size)
        self.h_compressed_idx_groupwise[row_idx, group_idx] = i
        self.h_compressed_mask_groupwise[row_idx, group_idx] = True
        self.h_compressed_sample[row_idx, self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]] = z_i.clone()
        self.h_compressed_mask[row_idx, self.h_group_start_index[group_idx]: self.h_group_end_index[group_idx]] = 1
        self.h_kl_beta[row_idx, group_idx] = 0

        return i, z_i

    def hh_compress_group(self, row_idx, group_idx):

        group_sample_size = int(np.ceil(2 ** self.bit_per_group))
        i, z_i, log_w = self.hh_sample_group(row_idx, group_idx, group_sample_size)
        self.hh_compressed_idx_groupwise[row_idx, group_idx] = i
        self.hh_compressed_mask_groupwise[row_idx, group_idx] = True
        self.hh_compressed_sample[row_idx, self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]] = z_i.clone()
        self.hh_compressed_mask[row_idx, self.hh_group_start_index[group_idx]: self.hh_group_end_index[group_idx]] = 1
        self.hh_kl_beta[row_idx, group_idx] = 0

        return i, z_i
    
    def train(self, x, y, n_epochs, optimizer, verbose, sample_size=5):
        for epoch in (tqdm(range(n_epochs)) if verbose else range(n_epochs)):
            y_pred = self.predict(x=x, random_seed=epoch, sample_size=sample_size)  # explicitly enforce reproducing
            if sample_size != 1:
                loss = torch.mean((y_pred - y[:, None, :, :]) ** 2) * y.shape[0]
            else:
                loss = torch.mean((y_pred - y) ** 2) * y.shape[0]
            elbo = loss
            kl = self.calculate_kl()
            elbo = elbo + kl
            if epoch % self.kl_adjust_gap == 0:
                self.update_annealing_factors(update=True)
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

    def optimize_posteriors(self, x, y, n_epochs, lr, verbose):
        if verbose:
            with torch.no_grad():
                y_pred = self.predict(x.to(self.device)).cpu()
                y_ori = y.cpu()
                distortion = np.mean(metric(y_ori.numpy(), y_pred.numpy(), self.dataset))
            print("Initialization: Average Distortion %.4f" % distortion, flush=True)
            if self.patch:
                kls, h_kls, hh_kls = self.update_annealing_factors(False)
                kl_bits = kls / np.log(2.)
                h_kl_bits = h_kls / np.log(2.)
                hh_kl_bits = hh_kls / np.log(2.)
                kl_bits = np.concatenate([kl_bits.reshape(-1), 
                                          h_kl_bits.reshape(-1), 
                                          hh_kl_bits.reshape(-1)])
            else:            
                kls = self.update_annealing_factors(False)
                kl_bits = kls / np.log(2.)
            print("Bits per group: ave %.2f" % kl_bits.mean() + " max %.2f" % kl_bits.max(), flush=True)
            print(' ')
            print("Start to optimize posteriors...", flush=True)

        optimizer = Adam(self.parameters(), lr=lr)
        self.train(x=x,  
                   y=y,  
                   n_epochs=n_epochs,
                   optimizer=optimizer,
                   verbose=verbose
                   )
        
        if verbose:
            with torch.no_grad():
                y_pred = self.predict(x.to(self.device)).cpu()
                y_ori = y.cpu()
                distortion = np.mean(metric(y_ori.numpy(), y_pred.numpy(), self.dataset))
            print("Optimization Finished. Average Distortion %.4f" % distortion, flush=True)
            if self.patch:
                kls, h_kls, hh_kls = self.update_annealing_factors(False)
                kl_bits = kls / np.log(2.)
                h_kl_bits = h_kls / np.log(2.)
                hh_kl_bits = hh_kls / np.log(2.)
                kl_bits = np.concatenate([kl_bits.reshape(-1), 
                                          h_kl_bits.reshape(-1), 
                                          hh_kl_bits.reshape(-1)])
            else:            
                kls = self.update_annealing_factors(False)
                kl_bits = kls / np.log(2.)

            print("Bits per group: ave %.2f" % kl_bits.mean() + " max %.2f" % kl_bits.max(), flush=True)

    def compress_posteriors(self,
                            x,
                            y,
                            n_epochs_finetune,
                            h_n_epochs_finetune,
                            hh_n_epochs_finetune,
                            verbose,
                            lr,
                            fine_tune_gap,
                            compress_from_group_with_largest_kl=True,
                            ):
        if verbose:
            print("Start to compress posteriors by A* coding...", flush=True)
        
        if self.patch:
            # compress the third level
            try:
                test = self.hh_compressed_num # save how many groups are compressed
            except:
                self.hh_compressed_num = 0
            hh_print_step = set(list(np.round(np.linspace(0, self.hh_n_groups, 10)).astype(int)))

            for _i in tqdm(range(self.hh_compressed_num, self.hh_n_groups)):
                for row_idx in range(self.hh_loc.shape[0]):
                    group_idx = _i
                    if compress_from_group_with_largest_kl:
                        kl_bits, h_kl_bits, hh_kl_bits = self.update_annealing_factors(False)
                        hh_kl_bits = hh_kl_bits[row_idx] / np.log(2.)
                        hh_mask = self.hh_compressed_mask_groupwise[row_idx]
                        hh_kl_bits[hh_mask] = -1e10
                        group_idx = hh_kl_bits.argmax()
                    self.hh_compress_group(row_idx, group_idx)
                    self.hh_compressed_num += 1

                if self.hh_compressed_num % fine_tune_gap == 0:
                    optimizer = Adam(self.parameters(), lr=lr) # reinitialize the momentums
                    self.train(x, 
                               y,
                               n_epochs=hh_n_epochs_finetune,
                               optimizer=optimizer,
                               verbose=False
                               )
                if _i in hh_print_step:
                    if verbose:
                        try:
                            with torch.no_grad():
                                y_pred = self.predict(x.to(self.device)).cpu()
                                y_ori = y.cpu()
                                distortion = np.mean(metric(y_ori.numpy(), y_pred.numpy(), self.dataset))

                            _, _, hh_kl_bits  = self.update_annealing_factors(False)
                            hh_kl_bits /= np.log(2.)
                            mask = (self.hh_compressed_mask_groupwise == False)
                            kl_bits = hh_kl_bits.flatten()[mask.flatten()]
                            print("Compress progress: %d" % (100 * self.hh_compressed_num / self.hh_n_groups),
                                "Average Distortion %.4f" % distortion, 
                                "KL in uncompressed groups: MAX %.3f" % kl_bits.max(), 
                                "AVE %.3f" % kl_bits.mean(), 
                                flush=True)
                        except:
                            pass
            if verbose:
                print(' ')

            # compress the second level
            try:
                test = self.h_compressed_num # save how many groups are compressed
            except:
                self.h_compressed_num = 0
            h_print_step = set(list(np.round(np.linspace(0, self.h_n_groups, 10)).astype(int)))

            for _i in tqdm(range(self.h_compressed_num, self.h_n_groups)):
                for row_idx in range(self.h_loc.shape[0]):
                    group_idx = _i
                    if compress_from_group_with_largest_kl:
                        kl_bits, h_kl_bits, hh_kl_bits = self.update_annealing_factors(False)
                        h_kl_bits = h_kl_bits[row_idx] / np.log(2.)
                        h_mask = self.h_compressed_mask_groupwise[row_idx]
                        h_kl_bits[h_mask] = -1e10
                        group_idx = h_kl_bits.argmax()
                    self.h_compress_group(row_idx, group_idx)
                    self.h_compressed_num += 1

                if self.h_compressed_num % fine_tune_gap == 0:
                    optimizer = Adam(self.parameters(), lr=lr) # reinitialize the momentums
                    self.train(x, 
                               y,
                               n_epochs=h_n_epochs_finetune,
                               optimizer=optimizer,
                               verbose=False
                               )
                if _i in h_print_step:
                    if verbose:
                        try:
                            with torch.no_grad():
                                y_pred = self.predict(x.to(self.device)).cpu()
                                y_ori = y.cpu()
                                distortion = np.mean(metric(y_ori.numpy(), y_pred.numpy(), self.dataset))

                            _, h_kl_bits, _  = self.update_annealing_factors(False)
                            h_kl_bits /= np.log(2.)
                            mask = (self.h_compressed_mask_groupwise == False)
                            kl_bits = h_kl_bits.flatten()[mask.flatten()]
                            print("Compress progress: %d" % (100 * self.h_compressed_num / self.h_n_groups),
                                "Average Distortion %.4f" % distortion, 
                                "KL in uncompressed groups: MAX %.3f" % kl_bits.max(), 
                                "AVE %.3f" % kl_bits.mean(), 
                                flush=True)
                        except:
                            pass
            if verbose:
                print(' ')

        try:
            test = self.compressed_num # save how many groups are compressed
        except:
            self.compressed_num = 0
        print_step = set(list(np.round(np.linspace(0, self.n_groups, 10)).astype(int)))

        for _i in tqdm(range(self.compressed_num, self.n_groups)):
            for row_idx in range(self.loc.shape[0]):
                group_idx = _i
                if compress_from_group_with_largest_kl:
                    if self.patch:
                        kl_bits, h_kl_bits, hh_kl_bits = self.update_annealing_factors(False)
                    else:
                        kl_bits = self.update_annealing_factors(False)
                    kl_bits = kl_bits[row_idx] / np.log(2.)
                    mask = self.compressed_mask_groupwise[row_idx]
                    kl_bits[mask] = -1e10
                    group_idx = kl_bits.argmax()
                self.compress_group(row_idx, group_idx)
            self.compressed_num += 1
            if self.compressed_num % fine_tune_gap == 0:
                optimizer = Adam(self.parameters(), lr=lr) # reinitialize the momentums
                self.train(x, 
                            y,
                            n_epochs=n_epochs_finetune,
                            optimizer=optimizer,
                            verbose=False
                            )
            if _i in print_step:
                if verbose:
                    try: 
                        with torch.no_grad():
                            y_pred = self.predict(x.to(self.device)).cpu()
                            y_ori = y.cpu()
                            distortion = np.mean(metric(y_ori.numpy(), y_pred.numpy(), self.dataset))
                        if self.patch:
                            kl_bits, _, _  = self.update_annealing_factors(False)
                        else:
                            kl_bits = self.update_annealing_factors(False)
                        kl_bits /= np.log(2.)
                        mask = (self.compressed_mask_groupwise == False)
                        kl_bits = kl_bits.flatten()[mask.flatten()]
                        print("Compress progress: %d; " % (100 * self.compressed_num / self.n_groups),
                                "Average Distortion %.4f; " % distortion, 
                                "KL in uncompressed groups: MAX %.3f" % kl_bits.max(), 
                                "AVE %.3f. " % kl_bits.mean(), 
                                flush=True)
                    except:
                        pass

        with torch.no_grad():
            y_pred = self.predict(x.to(self.device)).cpu()
            y_ori = y.cpu()
            distortion = metric(y_ori.numpy(), y_pred.numpy(), self.dataset)
        if verbose:
            print("Optimization Finished. Average Distortion %.4f" % np.mean(distortion), flush=True)
        return distortion















