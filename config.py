"""
Last updated on 2023-10-24

Args:
    input_dim: input dimension of the inr
    output_dim: output dimension of the inr
    hidden_dims: hidden dimensions of the inr

    data_dim: dimensonality of the data point (2 for image, 1 for audio, 3 for video, etc.)
    pixel_sizes: a list of pixel numbers along each dimension of the data. For image/video, it is set to the pixel size; for audio, it is set to the audio sample number; for protein, it is set to the number of residues.
    patch: patch or not
    hierarchical_patch_nums: If patching, it is a dict, showing how many patches are grouped together in each level. If not patching, set to None.
    patch_nums: a list of number of patches along each dimension.

    latent_dim: number of channels of the latent positional encodings
    paddings: padding parameter of each conv layer in the upsampling network
    scale_factors: scale factor of each upsampling layer in the upsampling network
    upsample_factors: upsampling factor along each dimension through the upsampling network
    
    bitrate_range: min_bitrate = max(max_bitrate - bitrate_range, lowest_bitrate)
    lowest_bitrate: min_bitrate = max(max_bitrate - bitrate_step, lowest_bitrate). These parameters are NOT very important. Can be set to any reasonable value
"""

{'cifar':
 {# INR parameters
  'input_dim': 32,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  # data and patch parameters
  'data_dim':2,
  'pixel_sizes': [32, 32],
  'patch': False,
  'hierarchical_patch_nums': None,
  'patch_nums':None,
  # positional encodings and upsampling network parameters
  'latent_dim':128,
  'paddings': [2, 1, 1],
  'scale_factors': [4, 2, 2], 
  'upsample_factors':[16, 16], 
  # other parameters
  'bitrate_range': 0.3,
  'lowest_bitrate': 0.1, 
  },

 'kodak':
 {# INR parameters
  'input_dim': 32,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  # data and patch parameters
  'data_dim':2,
  'patch': True,
  'pixel_sizes': [64, 64],
  'hierarchical_patch_nums': {'level2': [4, 4], 'level3': [8, 12]},
  'patch_nums': [512//64, 768//64],
  # positional encodings and upsampling network parameters
  'latent_dim':128,
  'paddings': [2, 1, 1],
  'scale_factors': [4, 2, 2], 
  'upsample_factors':[16, 16], 
  # other parameters
  'bitrate_range': 0.1,
  'lowest_bitrate': 0.05, 
  },

 'audio':
 {# INR parameters
  'input_dim': 32,
  'output_dim': 1,
  'hidden_dims': [32, ] * 3,
  # data and patch parameters
  'data_dim':1,
  'pixel_sizes': [800],
  'patch': True,
  'hierarchical_patch_nums': {'level2': [4], 'level3': [60]},
  'patch_nums': [48000//800],
  # positional encodings and upsampling network parameters
  'latent_dim':128,
  'paddings': [2, 1, 1],
  'scale_factors': [4, 2, 2], 
  'upsample_factors':[16], 
  # other parameters
  'bitrate_range': 0.3,
  'lowest_bitrate': 0.1, 
  },
  
 'video':
 {# INR parameters
  'input_dim': 34,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  # data and patch parameters
  'data_dim':3,
  'pixel_sizes': [16, 16, 24],
  'patch': True,
  'hierarchical_patch_nums': {'level2': [4, 4, 1], 'level3': [8, 8, 1]},
  'patch_nums': [128//16, 128//16, 24//24],
  # positional encodings and upsampling network parameters
  'latent_dim':128,
  'paddings': [2, 1, 1], 
  'scale_factors': [(6, 4, 4), 2, 2], 
  'upsample_factors':[16, 16, 24], 
  # other parameters
  'bitrate_range': 0.3,
  'lowest_bitrate': 0.1, 
  },

 'protein':
 {# INR parameters
  'input_dim': 32,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  # data and patch parameters
  'data_dim':1,
  'pixel_sizes': [96],
  'patch': False,
  'hierarchical_patch_nums': None,
  'patch_nums': None,
  # positional encodings and upsampling network parameters
  'latent_dim':128,
  'paddings': [2, 1, 1], 
  'scale_factors': [4, 2, 2], 
  'upsample_factors':[16], 
  # other parameters
  'bitrate_range': 0.3,
  'lowest_bitrate': 0.1, 
  },
}