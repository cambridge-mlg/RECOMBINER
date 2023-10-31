from test_model import *
from prior_model import *
from config import configs
from data.load_data import load_test_set

import argparse
import pickle
import os



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--test_idx', type=int, required=True) 
    parser.add_argument("--dataset", choices=("cifar", "kodak", "video", "audio", "protein"), )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prior_path", required=True)
    parser.add_argument("--save_dir", required=True)
    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse_args()
    config = configs[args.dataset]

    in_dim = config['input_dim']
    hidden_dims = config['hidden_dims']
    out_dim = config['output_dim']

    device = args.device

    # load priors
    with open(args.prior_path, "rb") as f:
        group_idx, group_start_index, group_end_index, group2param, param2group, n_groups, group_kls, weights = pickle.load(f)
        prior_loc, prior_scale, kl_beta, average_training_log_scale = pickle.load(f)
        h_group_idx, h_group_start_index, h_group_end_index, h_group2param, h_param2group, h_n_groups, h_group_kls, h_weights = pickle.load(f)
        h_prior_loc, h_prior_scale, _, h_average_training_log_scale = pickle.load(f)
        hh_group_idx, hh_group_start_index, hh_group_end_index, hh_group2param, hh_param2group, hh_n_groups, hh_group_kls, hh_weights = pickle.load(f)
        hh_prior_loc, hh_prior_scale, _, hh_average_training_log_scale = pickle.load(f)
        linear_transform = pickle.load(f)
        upsample_net = pickle.load(f)

    # load and reorder priors
    # the variable starting with _ is the reordered one
    p_locs = prior_loc.clone()
    _p_locs = p_locs[param2group].to(device)
    p_log_scales = torch.log(torch.exp(prior_scale * 6) - 1).clone()
    _p_log_scales = p_log_scales[param2group].to(device)
    
    if config['patch']:
        h_p_locs = h_prior_loc.clone()
        _h_p_locs = h_p_locs[h_param2group].to(device)
        h_p_log_scales = torch.log(torch.exp(h_prior_scale * 6) - 1).clone()
        _h_p_log_scales = h_p_log_scales[h_param2group].to(device)

        hh_p_locs = hh_prior_loc.clone()
        _hh_p_locs = hh_p_locs[hh_param2group].to(device)
        hh_p_log_scales = torch.log(torch.exp(hh_prior_scale * 6) - 1).clone()
        _hh_p_log_scales = hh_p_log_scales[hh_param2group].to(device)
    else:
        _h_p_locs = None
        _h_p_log_scales = None
        _hh_p_locs = None
        _hh_p_log_scales = None

    # load test data
    x, y = load_test_set(args.test_dir, 
                         args.test_idx,
                         args.dataset,
                         config['fourier_dim'],
                         config['patch'], 
                         config['patch_sizes']
                         )
    x = x.to(device)
    y = y.to(device)

    # initialize test model
    recombiner = TestBNNmodel(                 
        # network architectures and dataset info
        in_dim=in_dim,
        hidden_dims=hidden_dims,
        out_dim=out_dim,
        number_of_datapoints=x.shape[0],
        upsample_factors=config['upsample_factors'],
        latent_dim=config['latent_dim'],
        data_dim=config['data_dim'],
        pixel_sizes=config['pixel_sizes'],
        patch=config['patch'],
        patch_nums=config['patch_nums'], 
        hierarchical_patch_nums=config['hierarchical_patch_nums'],
        dataset=args.dataset,

        # learned mappings and priors
        linear_transform=linear_transform,
        upsample_net=upsample_net,

        p_loc=_p_locs,
        p_log_scale=_p_log_scales,
        init_log_scale=average_training_log_scale[param2group].cpu().detach(),
        param_to_group=param2group,
        group_to_param=group2param,
        n_groups=n_groups,
        group_start_index=group_start_index,
        group_end_index=group_end_index,
        group_idx=group_idx,

        h_p_loc=_h_p_locs,
        h_p_log_scale=_h_p_log_scales,
        h_init_log_scale=h_average_training_log_scale[h_param2group].cpu().detach(),
        h_param_to_group=h_param2group,
        h_group_to_param=h_group2param,
        h_n_groups=h_n_groups,
        h_group_start_index=h_group_start_index,
        h_group_end_index=h_group_end_index,
        h_group_idx=h_group_idx,

        hh_p_loc=_hh_p_locs,
        hh_p_log_scale=_hh_p_log_scales,
        hh_init_log_scale=hh_average_training_log_scale[hh_param2group].cpu().detach(),
        hh_param_to_group=hh_param2group,
        hh_group_to_param=hh_group2param,
        hh_n_groups=hh_n_groups,
        hh_group_start_index=hh_group_start_index,
        hh_group_end_index=hh_group_end_index,
        hh_group_idx=hh_group_idx,

        # other hyperparameters
        w0=30.,
        c=6.,
        random_seed=args.seed,
        device=device,
        kl_upper_buffer=0.,
        kl_lower_buffer=0.4,
        kl_adjust_gap=10,
        initial_beta=1e-8,
        beta_step_size=0.05
    ).to(device)

    recombiner.optimize_posteriors(x,
                                    y,
                                    n_epoch_kl=30000,
                                    lr=2e-4,
                                    verbose=1
                                    )
    distortion = recombiner.compress_posteriors(x,
                                                y,
                                                n_epochs_finetune=max(30000 // n_groups, 50),
                                                h_n_epochs_finetune=max(15000 // h_n_groups, 20),
                                                hh_n_epochs_finetune=max(15000 // hh_n_groups, 20),
                                                verbose=1,
                                                lr=2e-4,
                                                fine_tune_gap=1,
                                                compress_from_group_with_largest_kl=True)
    # save
    if isinstance(distortion, float):
        distortion = np.array([[distortion]])
    file_name = "Distortion_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(args.save_dir + file_name, distortion, delimiter=",")

    file_name = "GroupIndex_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(args.save_dir + file_name, recombiner.compressed_idx, delimiter=",")

    file_name = "H_GroupIndex_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(args.save_dir + file_name, recombiner.h_compressed_idx, delimiter=",")

    file_name = "HH_GroupIndex_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(args.save_dir + file_name, recombiner.hh_compressed_idx, delimiter=",")

if __name__ == '__main__':
    main()