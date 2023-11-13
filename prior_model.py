import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import kl_divergence, Normal

from utils import map_lpe_to_inr_inputs, \
                  map_hierarchical_model_to_int_weights, \
                  count_layer_params, \
                  count_net_params, \
                  PSNR, \
                  batch_PSNR

class LinearTransform(nn.Module):
    def __init__(self, net_dims):
        super().__init__()
        self.A = nn.ParameterList([(torch.rand(net_dims[i] * (net_dims[i-1]+1), 
                                               net_dims[i] * (net_dims[i-1]+1)) * 2 - 1) / (net_dims[i] * (net_dims[i-1]+1)) 
                                    for i in range(1, len(net_dims))])

class Upsample(nn.Module):
    def __init__(self, kernel_dim, paddings, layerwise_scale_factors):
        super().__init__()
        in_dim = 128
        hidden_dim = 64
        out_dim = 16
        self.up1 = nn.Upsample(scale_factor=layerwise_scale_factors[0])
        if kernel_dim == 1:
            self.conv1 = nn.Conv1d(in_dim, hidden_dim, 5, padding=paddings[0])
            self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=paddings[1])
            self.conv3 = nn.Conv1d(hidden_dim, out_dim, 3, padding=paddings[2])
        if kernel_dim == 2:
            self.conv1 = nn.Conv2d(in_dim, hidden_dim, 5, padding=paddings[0])
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=paddings[1])
            self.conv3 = nn.Conv2d(hidden_dim, out_dim, 3, padding=paddings[2])
        if kernel_dim == 3:
            self.conv1 = nn.Conv3d(in_dim, hidden_dim, 5, padding=paddings[0])
            self.conv2 = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=paddings[1])
            self.conv3 = nn.Conv3d(hidden_dim, out_dim, 3, padding=paddings[2])
        self.act1 = nn.LeakyReLU()
        self.up2 = nn.Upsample(scale_factor=layerwise_scale_factors[1])
        self.act2 = nn.LeakyReLU()
        self.up3 = nn.Upsample(scale_factor=layerwise_scale_factors[2])
        
    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.act1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.act2(x)

        x = self.up3(x)
        x = self.conv3(x)

        return x
    

class PriorBNNmodel(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dims,
                 out_dim,
                 train_size, # number of entire data * number of patches per data point
                 data_dim,
                 pixel_sizes,
                 upsample_factors,
                 latent_dim,
                 patch,
                 patch_nums, 
                 hierarchical_patch_nums,
                 random_seed=42,
                 device="cuda",
                 init_log_scale=-4,
                 c=6.,
                 w0=30.
                 ):
        super().__init__()
        self.random_seed = random_seed
        self.device = device
        self.n_layers = len(hidden_dims) + 1
        self.dims = [in_dim] + hidden_dims + [out_dim]
        self.patch = patch
        self.act = lambda x: torch.sin(w0 * x)
        self.st = lambda x: F.softplus(x, beta=1, threshold=20) / 6 # std transform fcn
        self.data_dim = data_dim
        self.train_size = train_size
        self.latent_dim = latent_dim
        self.pixel_sizes = pixel_sizes
        self.upsample_factors = upsample_factors
        self.patch_nums = patch_nums
        self.hierarchical_patch_nums = hierarchical_patch_nums

        # count the number of parameters per layer and their cumsum.
        self.net_params_list, self.cum_param_sizes = count_net_params(in_dim, hidden_dims, out_dim)
        
        torch.manual_seed(random_seed)
        w_std = np.sqrt(c / hidden_dims[-1]) / w0 # siren init (we found the first layer does not need to be treated specially)
        self.loc = nn.Parameter(torch.rand(train_size, self.cum_param_sizes[-1]) * w_std * 2 - w_std)
        self.log_scale = nn.Parameter(torch.zeros([train_size, self.cum_param_sizes[-1]]) + init_log_scale)
        if self.patch:
            self.h_loc = nn.Parameter(torch.rand(train_size//np.prod(hierarchical_patch_nums['level2']), self.cum_param_sizes[-1]) * w_std * 2 - w_std)
            self.h_log_scale = nn.Parameter(torch.zeros(train_size//np.prod(hierarchical_patch_nums['level2']), self.cum_param_sizes[-1]) + init_log_scale)
            self.hh_loc = nn.Parameter(torch.rand(train_size//np.prod(hierarchical_patch_nums['level3']), self.cum_param_sizes[-1]) * w_std * 2 - w_std)
            self.hh_log_scale = nn.Parameter(torch.zeros(train_size//np.prod(hierarchical_patch_nums['level3']), self.cum_param_sizes[-1]) + init_log_scale)
        self.lpe_loc = nn.Parameter(torch.randn(train_size, *[pixel_sizes[i]//upsample_factors[i] for i in range(data_dim)], latent_dim) * 0.1)
        self.lpe_log_scale = nn.Parameter(torch.zeros(train_size, *[pixel_sizes[i]//upsample_factors[i] for i in range(data_dim)], latent_dim) + init_log_scale)

    def group_to_layer(self, params, layer_idx):
        """
        map the entire INR weight vector to each layer
        """
        if layer_idx == 0:
            return params[..., :self.cum_param_sizes[layer_idx]]
        else:
            return params[..., self.cum_param_sizes[layer_idx - 1]: self.cum_param_sizes[layer_idx]]

    def layer_to_weight(self, in_dim, out_dim, layer_param):
        """
        map the INR weight vector of each layer into weight matrix and bias vector
        """
        bias = layer_param[:, :out_dim]
        weights = layer_param[:, out_dim:].reshape(-1, in_dim, out_dim)
        return weights, bias

    def forward(self, x, linear_transform, upsample_net, gradient_through_A=True):
        """
        map input coordinates to output pixel values.
        Args:
            x: (train_size, total_pixel_number, Fourier_embedding_dim)
            linear_transform: instance of LinearTransform class. 
            upsample_net: instance of Upsample class.
            gradient_through_A: if taking gradient through linear_transform.
        """
        assert x.shape[0] == self.train_size

        loc = self.loc
        scale = self.st(self.log_scale)

        lpe_loc = self.lpe_loc
        lpe_scale = self.st(self.lpe_log_scale)
        lpe = (lpe_loc + lpe_scale * torch.randn_like(lpe_loc))

        pe = map_lpe_to_inr_inputs(upsample_net,
                                   lpe[None, ...], # sample size = 1
                                   self.latent_dim,
                                   self.pixel_sizes,
                                   self.upsample_factors,
                                   self.patch,
                                   self.patch_nums,
                                   self.data_dim)[:, 0, ...] # pe shape (data_num, total_pixel_num, pe_dim)
        x = torch.cat([x,  pe], -1) # (data_num, total_pixel_num, pe_dim+ Fourier_embedding_dim)
        h_loc = self.h_loc if self.patch else None
        h_scale = self.st(self.h_log_scale) if self.patch else None
        hh_loc = self.hh_loc if self.patch else None
        hh_scale = self.st(self.hh_log_scale) if self.patch else None
        h_w = map_hierarchical_model_to_int_weights(use_hierarchical_model=self.patch,
                                                    loc=loc, scale=scale,
                                                    h_loc=h_loc, h_scale=h_scale,
                                                    hh_loc=hh_loc, hh_scale=hh_scale,
                                                    sample_size=1,
                                                    hierarchical_patch_nums=self.hierarchical_patch_nums,
                                                    patch_nums=self.patch_nums,
                                                    data_dim=self.data_dim)[:, 0, ...] # only have one sample, so squeeze the sample_size dimension
        for idx in range(self.n_layers):
            if gradient_through_A:
                A = linear_transform.A[idx]
            else:
                A = linear_transform.A[idx].detach()
            sample_latent = self.group_to_layer(h_w, idx)
            sample_latent = sample_latent @ A
            w, b = self.layer_to_weight(self.dims[idx], self.dims[idx + 1], sample_latent)
            x = (x @ w) + b[:, None, :]
            if idx != self.n_layers - 1:
                x = self.act(x)
        return x

    def calculate_kl(self,
                     prior_loc,
                     prior_scale,
                     prior_lpe_loc,
                     prior_lpe_scale,
                     prior_h_loc,
                     prior_h_scale,
                     prior_hh_loc,
                     prior_hh_scale,
                     ):
        kls = kl_divergence(Normal(self.loc, self.st(self.log_scale)),
                            Normal(prior_loc, prior_scale)).sum()
        kls += kl_divergence(Normal(self.lpe_loc, self.st(self.lpe_log_scale)),
                            Normal(prior_lpe_loc, prior_lpe_scale)).sum()
        if self.patch:
            kls += kl_divergence(Normal(self.h_loc, self.st(self.h_log_scale)),
                                Normal(prior_h_loc, prior_h_scale)).sum()
            kls += kl_divergence(Normal(self.hh_loc, self.st(self.hh_log_scale)),
                                Normal(prior_hh_loc, prior_hh_scale)).sum()
        return kls

    def train(self,
              n_epoch,
              lr,
              x,
              y,
              prior_loc,
              prior_scale,
              prior_lpe_loc,
              prior_lpe_scale,
              prior_h_loc,
              prior_h_scale,
              prior_hh_loc,
              prior_hh_scale,
              linear_transform, 
              upsample_net,
              kl_beta,
              training_mappings=True, # train upsampling net and linear transform or not
              verbose=False):
        
        x = x.to(self.device)
        y = y.to(self.device)

        if training_mappings:
            opt = Adam(list(self.parameters()) + list(linear_transform.parameters()) + list(upsample_net.parameters()), lr)
        else:
            opt = Adam(self.parameters(), lr)

        MSE = []
        ELBO = []
        for i in range(n_epoch) if not verbose else tqdm(range(n_epoch)):
            y_hat = self.forward(x,
                                 linear_transform, 
                                 upsample_net,
                                 gradient_through_A=training_mappings,
                                 )
            mse = torch.mean((y_hat - y) ** 2) * y.shape[0]  # number of images
            kl = self.calculate_kl(prior_loc,
                                   prior_scale,
                                   prior_lpe_loc,
                                   prior_lpe_scale,
                                   prior_h_loc,
                                   prior_h_scale,
                                   prior_hh_loc,
                                   prior_hh_scale,
                                   ) * kl_beta
            loss = mse + kl
            opt.zero_grad()
            loss.backward()
            opt.step()

            MSE.append(mse.item())
            ELBO.append(-loss.item())
        return MSE[-1] / y.shape[0], self.calculate_kl(prior_loc,
                                                        prior_scale,
                                                        prior_lpe_loc,
                                                        prior_lpe_scale,
                                                        prior_h_loc,
                                                        prior_h_scale,
                                                        prior_hh_loc,
                                                        prior_hh_scale,
                                                        ).item() / y.shape[0], ELBO

def get_grouping(q_loc, q_scale, prior_loc, prior_scale):
    """
    Assign parameters into groups so that sum of each group's KLs is close to but smaller than max_weight
    """
    kls = kl_divergence(Normal(q_loc, q_scale),
                        Normal(prior_loc, prior_scale))
    weights = (kls / np.log(2.)).mean(0).cpu().detach().numpy()
    return get_grouping_by_kl(weights)

def get_grouping_by_kl(kls_bits):
    """
    Assign parameters into groups so that sum of each group's KLs is close to but smaller than max_weight
    """
    parameters = np.arange(kls_bits.shape[0])
    weights = kls_bits
    np.random.seed(0)
    index = np.random.choice(weights.shape[0], weights.shape[0], False)
    np.random.seed(None)

    result = group_parameters(parameters[index], weights[index])
    n_groups = len(result)
    param2group = np.concatenate([np.array(i) for i in result])
    group2param = np.argsort(param2group)
    group_idx = np.concatenate([np.array([i, ] * len(result[i])) for i in range(len(result))]).astype(int)
    group_start_index = []
    group_end_index = []
    cursor = 0
    for i in result:
        group_start_index.append(cursor)
        cursor += len(i)
        group_end_index.append(cursor)
    group_start_index = np.array(group_start_index)
    group_end_index = np.array(group_end_index)

    group_kls = np.array([sum([weights[i] for i in group]) for group in result])
    return group_idx, group_start_index, group_end_index, group2param, param2group, n_groups, group_kls, weights

def group_parameters(parameters, weights, max_weight=16):
    """
    Assign parameters into groups so that sum of each group's weights is close to but smaller than max_weight
    """
    cursor = 1
    current_kl = weights[0]
    groups = [[parameters[0]]]
    for i in range(1, len(parameters)):
        if current_kl + weights[i] > max_weight:
            groups.append([parameters[i]])
            current_kl = weights[i]
        else:
            groups[-1].append(parameters[i])
            current_kl += weights[i]
        cursor += 1
    return groups
