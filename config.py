"""
Last updated on 2023-10-09

Args:
    bitrate_step: min_bitrate = max(max_bitrate - bitrate_step, lowest_bitrate)
    lowest_bitrate: min_bitrate = max(max_bitrate - bitrate_step, lowest_bitrate). These parameters are not very important. Can be set to any reasonable value
    input_dim: input dimension of the inr
    output_dim: output dimension of the inr
    hidden_dims: hidden dimensions of the inr
    patch: patch or not
    patch_size: patch size. None if not patching
    kernel_dim: kernel dimension of the upsampling network
    pixel_size: used to calculate the bitrate. for image/video, it is set to the pixel size; for audio, it is set to the audio sample number; for protein, it is set to the number of residues
"""



{'cifar':
 {'bitrate_step': 0.3,
  'lowest_bitrate': 0.1,   
  'input_dim': 32,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  'patch': False,
  'patch_size': None,
  'kernel_dim': 2,
  'pixel_size': 32**2,
  'paddings': [2, 1, 1],
  'hierarchical_patch_nums': None,
  'patch_sizes':None,
  'patch_nums':None,
  'args_for_upsampling_net':{'latent_dim':128,
                             'pixel_sizes':[32, 32], 
                             'upsample_factors':[16, 16], 
                             'patch':False, 
                             'data_dim':2
                             } 
  },

  'kodak':
 {'bitrate_step': 0.1,
  'lowest_bitrate': 0.05,   
  'input_dim': 32,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  'patch': True,
  'patch_size': 64,
  'kernel_dim': 2,
  'pixel_size': 64**2,
  'paddings': [2, 1, 1],
  'hierarchical_patch_nums': {'level2': [4, 4],
                              'level3': [8, 12]}, # how many patches does each representation in level2/3 contain
  'patch_sizes':[64, 64],
  'patch_nums':[512//64, 768//64],
  'args_for_upsampling_net':{'latent_dim':128,
                             'pixel_sizes':[64, 64], 
                             'upsample_factors':[16, 16], 
                             'patch':True, 
                             'data_dim':2
                             } 
  },
    
  'audio':
 {'bitrate_step': 0.3,
  'lowest_bitrate': 0.1,   
  'input_dim': 32,
  'output_dim': 1,
  'hidden_dims': [32, ] * 3,
  'patch': True,
  'patch_size': 800,
  'kernel_dim': 1,
  'pixel_size': 800,
  'paddings': [2, 1, 1],
  'hierarchical_patch_nums': {'level2': [4],
                              'level3': [60]},
  'patch_sizes':[800],
  'patch_nums':[48000//800],
  'args_for_upsampling_net':{'latent_dim':128,
                             'pixel_sizes':[800], 
                             'upsample_factors':[16], 
                             'patch':True, 
                             'data_dim':1
                             } 
  },
  
  'video':
 {'bitrate_step': 0.3,
  'lowest_bitrate': 0.1,   
  'input_dim': 34,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  'patch': True,
  'patch_size': 16,
  'kernel_dim': 3,
  'pixel_size': 16*16*24,
  'paddings': [2, 1, 1], # no!
  'hierarchical_patch_nums': {'level2': [4, 4, 1],
                              'level3': [8, 8, 1]},
  'patch_sizes':[16, 16, 24],
  'patch_nums':[128//16, 128//16, 24//24],
  'args_for_upsampling_net':{'latent_dim':128,
                             'pixel_sizes':[16, 16, 24], 
                             'upsample_factors':[16, 16, 24], 
                             'patch':True, 
                             'data_dim':3
                             } 
  },
  
  'protein':
 {'bitrate_step': 0.3,
  'lowest_bitrate': 0.1,   
  'input_dim': 32,
  'output_dim': 3,
  'hidden_dims': [32, ] * 3,
  'patch': False,
  'patch_size': None,
  'kernel_dim': 1,
  'pixel_size': 96,
  'paddings': [2, 1, 1],
  'hierarchical_patch_nums': None,
  'patch_sizes':None,
  'patch_nums':None,
  'args_for_upsampling_net':{'latent_dim':128,
                             'pixel_sizes':[96], 
                             'upsample_factors':[16], 
                             'patch':False, 
                             'data_dim':1
                             } 
  }}