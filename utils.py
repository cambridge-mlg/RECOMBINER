import numpy as np
import torch

def map_lpe_to_inr_inputs(upsample_net,
                          latent_pe,
                          latent_dim,
                          pixel_sizes,
                          upsample_factors,
                          patch,
                          patch_nums,
                          data_dim):
    """
    This function maps latent positional encodings (lpe) to the inputs of INRs.
    It first upsample lpe to upsampled coordinate-wise positional encodings, 
    and concatenate them with Fourier embeddings of coordinates.

    Note that if the data points are split into patches, the upsampling is not directly applied
    to each patch. Instead, the lpe for each patches are stitched together, and upsampled together.
    
    Args:
        upsample_net: the upsampling network
        latent_pe: torch tensor. latent positional encodings. shape (sample_size, data_num, latent_dim*product of spatial dimensions) or (sample_size, data_num, *spatial dimensions, latent_dim)
        latent_dim: latent positional encodings channels number
        pixel_sizes: A list of number of pixels along each dimension in each data point / patch.
        upsample_factors: upsampling factors when mapping latent positional encodings to upsampled positional encodings. List containing factor along each dimension
        patch: patch or not
        patch_nums: patch number along each dimension
        data_dim: data point dimensionality, e.g., 3 for video, 2 for image, 1 for audio
    """
    sample_size, data_num = latent_pe.shape[0:2]
    latent_pe_dims = [pixel_sizes[i] // upsample_factors[i] for i in range(data_dim)]

    # reshape latent_pe into spatial dimensions
    latent_pe = latent_pe.reshape(sample_size, 
                                  data_num,
                                  *latent_pe_dims,
                                  -1) # (sample_size, data_num, *pixel_sizes//*upsample_factors, latent_dim)
    assert latent_dim == latent_pe.shape[-1]

    # upsample the lpe to shape (sample_size, data_num, *pixel_sizes, n_channels)
    if patch == False:
        '''
        If not patch, we 
        1) permute latent_pe, so that channel dimension goes before spatial dimensions,
        2) input it through the upsampling network, 
        3) permute it back
        '''
        # permute latent_pe
        permute_index = [0, 1] + [len(latent_pe.shape)-1] + list(range(2, len(latent_pe.shape)-1))
        latent_pe = latent_pe.permute(permute_index)
        
        # upsample positional encodings
        latent_pe = latent_pe.reshape(-1, *latent_pe.shape[2:])
        pe = upsample_net(latent_pe)

        # permute back
        permute_index = [0] + list(range(2, len(pe.shape))) + [1]
        pe = pe.permute(permute_index)
        pe = pe.reshape(sample_size, data_num, *pe.shape[1:])
    else:
        '''
        If patch, we 
        1) assemble patches together
        2) permute latent_pe for each patch, so that channel dimension goes before spatial dimensions,
        3) input it through the upsampling network, 
        4) permute it back
        5) re-split upsampled positional encodings into patches
        Note, that if we treat patches separately as above, it will only have a very minor influence on performance.
        But it makes more sense to upsample them as a whole in principle. That is why we suggest this way.
        '''
        latent_pe = latent_pe.reshape(sample_size,
                                      -1, # number of entire data points
                                      *patch_nums,
                                      *latent_pe_dims,
                                      latent_dim)
        # assemble patches together
        # first, permute (*patch_nums, *latent_pe_dims) into (patch_nums[0], latent_pe_dims[0], patch_nums[1], latent_pe_dims[1], ...)
        permute_index_1 = [i + 2 for i in range(data_dim)] # index for *patch_nums
        permute_index_2 = [i + 2 for i in range(data_dim, data_dim * 2)] # index for *latent_pe_dims
        permute_index = []
        for i in range(data_dim):
           permute_index.append(permute_index_1[i])
           permute_index.append(permute_index_2[i])
        permute_index = [0, 1] + permute_index + [len(latent_pe.shape)-1]
        latent_pe = latent_pe.permute(permute_index)
        # second, aggregate each of (patch_nums[i], latent_pe_dims[i]) together
        latent_pe = latent_pe.reshape(sample_size, 
                                      -1, 
                                      *[patch_nums[i]*pixel_sizes[i]//upsample_factors[i] for i in range(data_dim)],
                                      latent_dim)
        
        # permute latent_pe's channel
        permute_index = [0, 1] + [len(latent_pe.shape)-1] + list(range(2, len(latent_pe.shape)-1))
        latent_pe = latent_pe.permute(permute_index)
        
        # upsample positional encodings
        latent_pe = latent_pe.reshape(-1, *latent_pe.shape[2:])
        pe = upsample_net(latent_pe)

        # permute the channel back
        permute_index = [0] + list(range(2, len(pe.shape))) + [1]
        pe = pe.permute(permute_index)

        # re-split pe into patches
        # first, reshape each spatial dimension into (patch_nums[i], patch_sizes[i])
        shapes = []
        for i in range(data_dim):
            shapes.append(patch_nums[i])
            shapes.append(pixel_sizes[i])
        pe = pe.reshape(sample_size, -1, *shapes, pe.shape[-1])
        # second, permute (patch_nums[0], patch_sizes[0], patch_nums[1], patch_sizes[1], ...) into (*patch_nums, *patch_sizes)
        permute_index_1 = [2 * i + 2 for i in range(data_dim)] # index for patch_nums
        permute_index_2 = [2 * i + 3 for i in range(data_dim)] # index for patch_sizes
        permute_index = permute_index_1 + permute_index_2
        permute_index = [0, 1] + permute_index + [len(pe.shape)-1]
        pe = pe.permute(permute_index)

    # reshape pe to data_num, sample_size, total_pixel_num, pe_dim
    pe = pe.reshape(sample_size, data_num, -1, pe.shape[-1]).permute(1, 0, 2, 3)
    return pe

def map_hierarchical_model_to_int_weights(use_hierarchical_model,
                                          loc, scale,
                                          h_loc, h_scale,
                                          hh_loc, hh_scale,
                                          sample_size,
                                          hierarchical_patch_nums,
                                          patch_nums,
                                          data_dim,
                                          ):
    """
    This function samples hierarchical model to inr weights (before linear transform, i.e., h_w)

    Args:
        use_hierarchical_model: bool. use hierarchical model or not
        loc, scale, h_loc, h_scale, hh_loc, hh_scale: mean and std of variables in each level
        sample_size: sample size
        hierarchical_patch_nums: dist. how many patches does each group in level 2/3 contain
        patch_nums: list. patch num along each dimension
        data_dim: data point dimensionality, e.g., 3 for video, 2 for image, 1 for audio
    """
    if use_hierarchical_model:
        data_num = loc.shape[0]

        # sample level 1
        loc = loc[:, None, :]
        scale = scale[:, None, :].repeat([1, sample_size, 1])
        sample_h_w = loc + scale * torch.randn_like(scale)

        # reshape and repeat level 2
        number_of_groups = [patch_nums[i]//hierarchical_patch_nums['level2'][i] for i in range(data_dim)]
        h_loc = h_loc.reshape(-1, *number_of_groups, h_loc.shape[-1])
        h_scale = h_scale.reshape(-1, *number_of_groups, h_scale.shape[-1])
        if data_dim == 1:
            h_loc = h_loc[:, :, None, :].repeat([1, 1, hierarchical_patch_nums['level2'][0], 1])
            h_scale = h_scale[:, :, None, :].repeat([1, 1, hierarchical_patch_nums['level2'][0], 1])
        elif data_dim == 2:
            h_loc = h_loc[:, :, None, :, None, :].repeat([1, 
                                                          1, 
                                                          hierarchical_patch_nums['level2'][0], 1, 
                                                          hierarchical_patch_nums['level2'][1], 1])
            h_scale = h_scale[:, :, None, :, None, :].repeat([1,
                                                              1, 
                                                              hierarchical_patch_nums['level2'][0], 1, 
                                                              hierarchical_patch_nums['level2'][1], 1])
        elif data_dim == 3:
            h_loc = h_loc[:, :, None, :, None, :, None, :].repeat([1, 
                                                                   1, 
                                                                   hierarchical_patch_nums['level2'][0], 1, 
                                                                   hierarchical_patch_nums['level2'][1], 1,
                                                                   hierarchical_patch_nums['level2'][2], 1])
            h_scale = h_scale[:, :, None, :, None, :, None, :].repeat([1, 
                                                                       1, 
                                                                       hierarchical_patch_nums['level2'][0], 1, 
                                                                       hierarchical_patch_nums['level2'][1], 1,
                                                                       hierarchical_patch_nums['level2'][2], 1])      
        else:
            raise NotImplementedError
        h_loc = h_loc.reshape([-1, h_loc.shape[-1]])[:, None, :]
        h_scale = h_scale.reshape([-1, h_loc.shape[-1]])[:, None, :].repeat([1, sample_size, 1])
        h = h_loc + torch.randn_like(h_scale) * h_scale

        # reshape and repeat level 3. Note, that level 3 only have a global representation   
        hh_loc = hh_loc[:, None, :].repeat([1, np.prod(patch_nums), 1]).reshape(-1, hh_loc.shape[-1])
        hh_scale = hh_scale[:, None, :].repeat([1, np.prod(patch_nums), 1]).reshape(-1, hh_scale.shape[-1])
        
        hh_loc = hh_loc[:, None, :] # 1, sample_size, dim
        hh_scale = hh_scale[:, None, :].repeat([1, sample_size, 1])  # 1, sample_size, dim
        hh = hh_loc + hh_scale * torch.randn_like(hh_scale)

        sample_h_w = sample_h_w + h + hh

    else:
        loc = loc[:, None, :]
        scale = scale[:, None, :].repeat([1, sample_size, 1])
        sample_h_w = loc + scale * torch.randn_like(scale)
    
    return sample_h_w

def metric(original, compressed, dataset):
    """
    This function calculate the metric results. This function will be called when compressing a test datapoint.
    """
    if dataset == 'cifar':
        return batch_PSNR(original, compressed, round=True, max_value=1)
    if dataset == 'kodak':
        return PSNR(original, compressed, round=True, max_value=1)
    if dataset == 'video':
        return PSNR(original, compressed, round=True, max_value=1)
    if dataset == 'audio':
        return PSNR(original, compressed, round=False, max_value=1)
    if dataset == 'protein':
        return batch_RMSD(original, compressed, scale_factor=25)
    
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


def PSNR(original, compressed, round, max_value=1):
    """
    Calculate PSNR of one image, or patches of one image.
    """
    if round:
        compressed = np.round(np.clip(compressed, 0, 1) * 255) / 255
    mse = np.mean((original - compressed) ** 2)
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr.item()


def batch_PSNR(original, compressed, round, max_value=1):
    """
    Calculate PSNR of images in one batch
    """
    batch_size = original.shape[0]
    if round:
        compressed = np.round(np.clip(compressed, 0, 1) * 255) / 255
    mse = np.mean((original.reshape(batch_size, -1) - compressed.reshape(batch_size, -1)) ** 2, axis=-1)
    psnr = 20 * np.log10(max_value / np.sqrt(mse))
    return psnr


def batch_RMSD(original, compressed, scale_factor):
    batch_size = original.shape[0]
    rmsd = (((original * scale_factor - compressed * scale_factor) ** 2).reshape(batch_size, -1).mean(-1) * 3) ** 0.5
    return rmsd


# Below are function modified from COMBINER (https://github.com/cambridge-mlg/combiner)
# These functions process data into Fourier embedded coords and signal values
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


def to_grid_coordinates_and_features(datum):
    """Converts a datum to a set of coordinates and features.
    Args:
        datum (torch.Tensor): Shape (channels, *spatial_dims).
    """
    # Coordinates are indices of all non-zero locations of a tensor of ones of
    # same shape as spatial dimensions of image

    coordinates = make_coord_grid(datum.shape[1:], (-1, 1), device=datum.device).view(-1, len(datum.shape[1:]))
    # Convert image to a tensor of features of shape (num_points, channels)
    features = datum.reshape(datum.shape[0], -1).T
    return coordinates, features