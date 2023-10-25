from utils import *

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import kl_divergence, Normal


class Mapping(nn.Module):
    def __init__(self, net_dims):
        super().__init__()
        self.A = nn.ParameterList([(torch.rand(net_dims[i] * (net_dims[i-1]+1), net_dims[i] * (net_dims[i-1]+1)) * 2 - 1) / (net_dims[i] * (net_dims[i-1]+1)) for i in range(1, len(net_dims))])

class Upsample(nn.Module):
    def __init__(self, kernel_dim, paddings, upsample_factors):
        super().__init__()
        in_dim = 128
        hidden_dim = 64
        out_dim = 16
        self.up1 = nn.Upsample(scale_factor=upsample_factors[0])
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
        self.up2 = nn.Upsample(scale_factor=upsample_factors[1])
        self.act2 = nn.LeakyReLU()
        self.up3 = nn.Upsample(scale_factor=upsample_factors[2])
        
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
                 training_set_size,
                 pixel_size,
                 upsample_factor,
                 latent_dim,
                 patch,
                 random_seed=42,
                 device="cuda",
                 init_log_scale=-4,
                 c=6.,
                 w0=30.,
                 kl_buffer=.5,
                 kl_budget=16,
                 ):
        super().__init__()
        self.n_layers = len(hidden_dims) + 1
        self.dims = [in_dim] + hidden_dims + [out_dim]
        self.device = device
        self.random_seed = random_seed

        self.patch = patch

        self.act = lambda x: torch.sin(w0 * x)
        self.st = lambda x: F.softplus(x, beta=1, threshold=20) / 6

        self.net_params_list, self.cum_param_sizes = count_net_params(in_dim, hidden_dims, out_dim)
        w_std = np.sqrt(c / hidden_dims[-1]) / w0

        torch.manual_seed(random_seed)
        if self.patch:
            # randomly initialize all training instances' loc
            self.loc = nn.Parameter(torch.rand(training_set_size, self.cum_param_sizes[-1]) * w_std * 2 - w_std)
            self.log_scale = nn.Parameter(torch.zeros([training_set_size, self.cum_param_sizes[-1]]) + init_log_scale)
            self.hyper_loc_loc = nn.Parameter(torch.rand(training_set_size//16, self.cum_param_sizes[-1]) * w_std * 2 - w_std)
            self.hyper_loc_log_scale = nn.Parameter(torch.zeros(training_set_size//16, self.cum_param_sizes[-1]) + init_log_scale)
            self.hh_loc_loc = nn.Parameter(torch.rand(training_set_size//96, self.cum_param_sizes[-1]) * w_std * 2 - w_std)
            self.hh_loc_log_scale = nn.Parameter(torch.zeros(training_set_size//96, self.cum_param_sizes[-1]) + init_log_scale)

        self.c_loc = nn.Parameter(torch.randn(training_set_size, pixel_size//upsample_factor, pixel_size//upsample_factor, latent_dim) * 0.1)
        self.c_log_scale = nn.Parameter(torch.zeros(training_set_size, pixel_size//upsample_factor, pixel_size//upsample_factor, latent_dim) + init_log_scale)

        self.kl_buffer = kl_buffer
        self.kl_budget = kl_budget

    def group_to_layer(self, params, layer_idx):
        if layer_idx == 0:
            return params[..., :self.cum_param_sizes[layer_idx]]
        else:
            return params[..., self.cum_param_sizes[layer_idx - 1]: self.cum_param_sizes[layer_idx]]

    def layer_to_weight(self, in_dim, out_dim, layer_param):
        bias = layer_param[:, :out_dim]
        weights = layer_param[:, out_dim:].reshape(-1, in_dim, out_dim)
        return weights, bias

    def forward(self, x, mapping, coord_mapping, gradient_through_A=True):

        loc = self.loc
        scale = self.st(self.log_scale)
        c_loc = self.c_loc
        c_scale = self.st(self.c_log_scale)
        coord_feature = (c_loc + c_scale * torch.randn_like(c_loc)) # n_patches, H, W, C
        if self.c_loc.shape[1] == 4: # if kodak
            coord_feature = coord_feature.reshape(-1, 96, 4, 4, 128)
            coord_feature = coord_feature.reshape(-1, 512//64, 768//64, 4, 4, 128)
            coord_feature = coord_feature.permute([0, 1, 3, 2, 4, 5])
            coord_feature = coord_feature.reshape([-1, 512//16, 768//16, 128]).permute([0, 3, 1, 2])

            up_coord_feature = []
            bsize = 84
            bnumb = int(np.ceil(coord_feature.shape[0]  / 84))
            for b in range(bnumb):
                up_coord_feature.append(coord_mapping(coord_feature[b * bsize: b * bsize + bsize]))
            coord_feature = torch.cat(up_coord_feature, 0)

            coord_feature = coord_feature.permute([0, 2, 3, 1]) # if kodak (n_image, 512, 768, D)
            coord_feature = coord_feature.reshape(-1, 512//64, 64, 768//64, 64, coord_feature.shape[-1])
            coord_feature = coord_feature.permute([0, 1, 3, 2, 4, 5])
            coord_feature = coord_feature.reshape([-1, 96, 64, 64, coord_feature.shape[-1]]).reshape([-1, 64, 64, coord_feature.shape[-1]])
        else:
            coord_feature = coord_mapping(coord_feature.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])
        coord_feature  = coord_feature.reshape([
            coord_feature.shape[0],
            -1,
            coord_feature.shape[-1]
        ])
        x = torch.cat([x, coord_feature], -1) # n_patches, n_pixels, n_dims

        if 
            # if kodak, loc and scale are determined by their hyper-priors
            hyper_loc_loc = self.hyper_loc_loc.reshape([-1, 6, self.hyper_loc_loc.shape[-1]]).reshape([-1, 2, 3, self.hyper_loc_loc.shape[-1]]) # n_images, 2, 3, dim
            hyper_loc_loc = hyper_loc_loc[:, :, None, :, None, :] # n_images, 2, 1, 3, 1, dim
            hyper_loc_loc = hyper_loc_loc.repeat([1, 1, 4, 1, 4, 1]) # n_images, 2, 4, 3, 4, dim
            hyper_loc_loc = hyper_loc_loc.reshape([-1, 96, self.hyper_loc_loc.shape[-1]]).reshape([-1, self.hyper_loc_loc.shape[-1]])
            
            hyper_loc_scale = self.st(self.hyper_loc_log_scale).reshape([-1, 6, self.hyper_loc_loc.shape[-1]]).reshape([-1, 2, 3, self.hyper_loc_loc.shape[-1]]) # n_images, 2, 3, dim
            hyper_loc_scale = hyper_loc_scale[:, :, None, :, None, :] 
            hyper_loc_scale = hyper_loc_scale.repeat([1, 1, 4, 1, 4, 1]) # n_images, 2, 4, 3, 4, dim
            hyper_loc_scale = hyper_loc_scale.reshape([-1, 96, self.hyper_loc_loc.shape[-1]]).reshape([-1, self.hyper_loc_loc.shape[-1]])

            hyper_loc = hyper_loc_loc + torch.randn_like(hyper_loc_scale) * hyper_loc_scale

            hh_loc_loc = self.hh_loc_loc[:, None, :].repeat([1, 96, 1]).reshape([-1, self.hh_loc_loc.shape[-1]])
            hh_loc_scale = self.st(self.hh_loc_log_scale)[:, None, :].repeat([1, 96, 1]).reshape([-1, self.hh_loc_loc.shape[-1]])

            hh_loc = hh_loc_loc + torch.randn_like(hh_loc_scale) * hh_loc_scale
            
            sample_latent_all = loc + torch.randn_like(scale) * scale
            sample_latent_all = sample_latent_all + hyper_loc + hh_loc

        for idx in range(self.n_layers):
            if gradient_through_A:
                A = mapping.A[idx]
            else:
                A = mapping.A[idx].detach()

            sample_latent = self.group_to_layer(sample_latent_all, idx)
            sample_latent = sample_latent @ A

            w, b = self.layer_to_weight(self.dims[idx], self.dims[idx + 1], sample_latent)

            x = (x @ w) + b[:, None, :]

            if idx != self.n_layers - 1:
                x = self.act(x)
        return x

    def calculate_kl(self,
                     prior_loc,
                     prior_scale,
                     c_prior_loc,
                     c_prior_scale,
                     h_loc_prior_loc,
                     h_loc_prior_scale,
                     hh_loc_prior_loc,
                     hh_loc_prior_scale,
                     ):
        kls = kl_divergence(Normal(self.loc, self.st(self.log_scale)),
                            Normal(prior_loc[None, :], prior_scale[None, :])).sum()
        kls += kl_divergence(Normal(self.c_loc, self.st(self.c_log_scale)),
                            Normal(c_prior_loc, c_prior_scale)).sum()
        kls += kl_divergence(Normal(self.hyper_loc_loc, self.st(self.hyper_loc_log_scale)),
                             Normal(h_loc_prior_loc, h_loc_prior_scale)).sum()
        kls += kl_divergence(Normal(self.hh_loc_loc, self.st(self.hh_loc_log_scale)),
                             Normal(hh_loc_prior_loc, hh_loc_prior_scale)).sum()

        return kls

    def calculate_kl_with_beta(self,
                               prior_loc,
                               prior_scale,
                               c_prior_loc,
                               c_prior_scale,
                               h_loc_prior_loc,
                               h_loc_prior_scale,
                               hh_loc_prior_loc,
                               hh_loc_prior_scale,
                               beta):
        kls = self.calculate_kl(prior_loc,
                                prior_scale,
                                c_prior_loc,
                                c_prior_scale,
                                h_loc_prior_loc,
                                h_loc_prior_scale,
                                hh_loc_prior_loc,
                                hh_loc_prior_scale,)
        return kls * beta


    def train(self,
              n_epoch,
              batch_size,
              lr,
              x,
              y,
              prior_loc,
              prior_scale,
              c_prior_loc,
              c_prior_scale,
              h_loc_prior_loc,
              h_loc_prior_scale,
              hh_loc_prior_loc,
              hh_loc_prior_scale,
              mapping,
              coord_mapping,
              training_mapping,
              kl_beta,
              verbose=False):
        if training_mapping:
            opt = Adam(list(self.parameters()) + list(mapping.parameters()) + list(coord_mapping.parameters()), lr)
        else:
            opt = Adam(self.parameters(), lr)
        for i in range(n_epoch) if not verbose else tqdm(range(n_epoch)):

            MSE = []
            batch_x = x.to(self.device)
            batch_y = y.to(self.device)
            y_hat = self.forward(batch_x,
                                    mapping,
                                    coord_mapping,
                                    gradient_through_A=training_mapping,
                                    )
            mse = torch.mean((y_hat - batch_y) ** 2) * y.shape[0]  # number of images
            kl = self.calculate_kl_with_beta(prior_loc,
                                             prior_scale,
                                             c_prior_loc,
                                             c_prior_scale,
                                             h_loc_prior_loc,
                                             h_loc_prior_scale,
                                             hh_loc_prior_loc,
                                             hh_loc_prior_scale,
                                             beta=kl_beta,
                                             )
            loss = mse + kl
            opt.zero_grad()
            loss.backward()
            opt.step()

            MSE.append(mse.item())
        return np.mean(MSE) / y.shape[0], self.calculate_kl(prior_loc,
                                                            prior_scale,
                                                            c_prior_loc,
                                                            c_prior_scale,
                                                            h_loc_prior_loc,
                                                            h_loc_prior_scale,
                                                            hh_loc_prior_loc,
                                                            hh_loc_prior_scale,
                                                            ).item()


def group_parameters(parameters, weights, max_weight=16):
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

def get_grouping(q_loc, q_scale, prior_loc, prior_scale):
    kls = kl_divergence(Normal(q_loc, q_scale),
                        Normal(prior_loc, prior_scale))
    weights = (kls / np.log(2.)).mean(0).cpu().detach().numpy()
    return get_grouping_by_kl(weights)

def get_grouping_by_kl(kls_bits):
    parameters = np.arange(kls_bits.shape[0])
    weights = kls_bits
    np.random.seed(0)
    index = np.random.choice(weights.shape[0], weights.shape[0], False)
    np.random.seed(None)

    result = group_parameters(parameters[index], weights[index])
    n_groups = len(result)
    param2group = np.concatenate([np.array(i) for i in result])
    group2param = np.argsort(param2group)
    group_idx = np.concatenate([np.array([i, ] * len(result[i])) for i in range(len(result))])
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



