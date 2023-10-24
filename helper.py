import numpy as np
import torch

def map_input_to_inr(upsample_net,
                     latent_pe, 
                     latent_dim,
                     pixel_sizes, 
                     upsample_factors, 
                     patch,
                     patch_nums,
                     data_dim,
                     ):
    """
    This function map latent positional encodings to upsampled positional encodings, 
    and concatenate them with Fourier embeddings of coordinates as the input to INRs.

    Args:
        upsample_net: upsampling network
        latent_pe: latent positional encodings' samples, shape (sample_size, data_number, latent_dim*product of spatial dimensions)
        latent_dim: latent positional encodings channels number
        pixel_size: pixel sizes for each data point / patch. List containing size along each dimension
        upsample_factors: upsampling factors when mapping latent positional encodings to upsampled positional encodings. List containing factor along each dimension
        patch: patch or not
        patch_nums: patch number along each dimension
        data_dim: data point dimensionality, e.g., 3 for video, 2 for image, 1 for audio
    """
    sample_size = latent_pe.shape[0]
    data_num = latent_pe.shape[1]
    latent_pe_dims = [pixel_sizes[i] // upsample_factors[i] for i in range(data_dim)]
    latent_pe = latent_pe.reshape(sample_size, 
                                  data_num,
                                  *latent_pe_dims,
                                  latent_dim)
    if patch == False:
        # permute latent_pe
        permute_index = [0, 1] + [len(latent_pe.shape)-1] + list(range(2, len(latent_pe.shape)-1))
        latent_pe = latent_pe.permute(permute_index)
        
        # upsample positional encodings
        latent_pe = latent_pe.reshape(-1, *latent_pe.shape[2:])
        pe = upsample_net(latent_pe)

        # permute back
        permute_index = [0] + list(range(2, len(pe.shape))) + [1]

        # reshape pe to data_num, sample_size, pixel_num, pe_dim
        pe = pe.permute(permute_index).reshape(sample_size, data_num, *pe.shape[1:])
        pe = pe.reshape(sample_size, data_num, -1, pe.shape[-1]).permute(1, 0, 2, 3)

    else:
        # assemble latent_pe for patches together
        latent_pe = latent_pe.reshape(sample_size,
                                      -1, # entire data num
                                      *patch_nums,
                                      *latent_pe_dims,
                                      latent_dim)
        permute_index_1 = [i + 2 for i in range(data_dim)]
        permute_index_2 = [i + 2 for i in range(data_dim, data_dim * 2)] 
        permute_index = []
        for i in range(data_dim):
           permute_index.append(permute_index_1[i])
           permute_index.append(permute_index_2[i])
        permute_index = [0, 1] + permute_index + [len(latent_pe.shape)-1]
        latent_pe = latent_pe.permute(permute_index)
        latent_pe = latent_pe.reshape(sample_size, 
                                      -1, # entire data num
                                      *[patch_nums[i]*pixel_sizes[i]//upsample_factors[i] for i in range(data_dim)],
                                      latent_dim)
        
        # permute latent_pe
        permute_index = [0, 1] + [len(latent_pe.shape)-1] + list(range(2, len(latent_pe.shape)-1))
        latent_pe = latent_pe.permute(permute_index)
        
        
        # upsample positional encodings
        latent_pe = latent_pe.reshape(-1, *latent_pe.shape[2:])
        pe = upsample_net(latent_pe)

        # permute back
        permute_index = [0] + list(range(2, len(pe.shape))) + [1]
        pe = pe.permute(permute_index)

        # reshape and split into patches
        shapes = []
        for i in range(data_dim):
            shapes.append(patch_nums[i])
            shapes.append(pixel_sizes[i])
        pe = pe.reshape(sample_size, 
                        -1,
                        *shapes,
                        pe.shape[-1])
        # put patch dimension and patch number dimension together respectively
        permute_index_1 = [i + 2 for i in range(data_dim)]
        permute_index_2 = [i + 2 for i in range(data_dim, data_dim * 2)] 
        permute_index = []
        for i in range(data_dim):
           permute_index.append(permute_index_1[i])
           permute_index.append(permute_index_2[i])
        permute_index = [0, 1] + permute_index + [len(latent_pe.shape)-1]
        pe = pe.permute(permute_index)
        pe = pe.reshape(sample_size, data_num, -1, pe.shape[-1]).permute(1, 0, 2, 3)
    return pe

def map_hierarchical_model_to_inr_weights(hierarchical_model,
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
        hierarchical_model: bool. use hierarchical model or not
        hierarchical_patch_nums: dist. how many patches does each group in level 2/3 contain
        patch_nums: list. patch num along each dimension


    """
    if hierarchical_model:
        data_num = loc.shape[0]

        # sample level 1
        loc = loc[:, None, :]
        scale = scale[:, None, :].repeat([1, sample_size, 1])
        sample_latent_all = loc + scale * torch.randn_like(scale)

        # reshape and repeat level 2
        number_of_groups = [patch_nums[i]//hierarchical_patch_nums['level2'][i] for i in range(data_dim)]
        h_loc = h_loc.reshape(data_num, *number_of_groups, -1)
        h_scale = h_scale.reshape(data_num, *number_of_groups, -1)
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
        h_scale = h_scale.reshape([-1, h_loc.shape[-1]])[:, None, :].reshape([1, sample_size, 1])
        h = h_loc + torch.randn_like(h_scale) * h_scale

        # reshape and repeat level 3. Note, that level 3 only have a global representation   
        hh_loc = hh_loc[:, None, :].repeat([1, np.prod(patch_nums), 1]).reshape(-1, hh_loc.shape[-1])
        hh_scale = hh_scale[:, None, :].repeat([1, np.prod(patch_nums), 1]).reshape(-1, hh_scale.shape[-1])
        
        hh_loc = hh_loc[:, None, :] # 1, sample_size, dim
        hh_scale = hh_scale[:, None, :].repeat([1, sample_size, 1])  # 1, sample_size, dim
        hh = hh_loc + hh_scale * torch.randn_like(hh_scale)

        sample_latent_all = sample_latent_all + h + hh

    else:
        loc = loc[:, None, :]
        scale = scale[:, None, :].repeat([1, sample_size, 1])
        sample_latent_all = loc + scale * torch.randn_like(scale)
    
    return sample_latent_all



