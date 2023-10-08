"""
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
  }}