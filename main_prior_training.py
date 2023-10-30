from prior_model import *
from config import configs
from data.load_data import load_training_set

import torch
import argparse
import os
import pickle
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--train_dir', required=True, help='training dir')
    parser.add_argument('--train_size', type=int, default=10000000000, help='training size. Default choice is to use all dataset in train_dir. Note, that if patches are used, please specific the patch number here. If the total number specified here is larger than the total available number, all instances will be used.')
    parser.add_argument("--dataset", choices=("cifar", "kodak", "video", "audio", "protein"), )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_bitrate", type=float, required=True, help="maximum bitrate budget (bpp or kbps or bits per residue)")
    parser.add_argument("--saving_dir", default="./", help="dir to save prior/linear transform/upsampling net/initializations")
    args = parser.parse_args()
    return args



def main():
    # parse arguments
    args = parse_args()
    config = configs[args.dataset]

    in_dim = config['input_dim']
    hidden_dims = config['hidden_dims']
    out_dim = config['output_dim']


    if config['patch']:
        number_of_entire_training_instances = args.train_size // np.prod(config['patch_nums'])
    else:
        number_of_entire_training_instances = args.train_size
    X, Y = load_training_set(args.train_dir, 
                             args.dataset, 
                             args.seed,
                             number_of_entire_training_instances,
                             config['fourier_dim'],
                             config['patch'], 
                             config['patch_sizes'])
    X = X.to(args.device)
    Y = Y.to(args.device)
    train_size = X.shape[0]
    print("Prior is trained on %d patches/images." % train_size, flush=True)

    # defined model and mappings
    device = args.device
    prior_model = PriorBNNmodel(in_dim=in_dim,
                                hidden_dims=hidden_dims,
                                out_dim=out_dim,
                                train_size=train_size,
                                data_dim=config['data_dim'],
                                pixel_sizes=config['pixel_sizes'],
                                upsample_factors=config['upsample_factors'],
                                latent_dim=config['latent_dim'],
                                patch=config['patch'],
                                patch_nums=config['patch_nums'], 
                                hierarchical_patch_nums=config['hierarchical_patch_nums'],
                                random_seed=args.seed,
                                device=device,
                                init_log_scale=-4,
                                c=6.,
                                w0=30.
                                ).to(device)
    linear_transform = LinearTransform(prior_model.dims).to(device)
    upsample_net = Upsample(kernel_dim=config['data_dim'],
                             paddings=config['paddings'], 
                             layerwise_scale_factors=config['layerwise_scale_factors']).to(device)

    kl_beta = 1e-8 # initial beta 
    budget_max = args.max_bitrate * np.prod(config['pixel_sizes']) 
    budget_min = max(config['lowest_bitrate'], (args.max_bitrate - config['bitrate_range'])) * np.prod(config['pixel_sizes']) 
    assert budget_min <= budget_max

    # initialize priors
    prior_loc = torch.zeros(prior_model.loc.shape[1]).to(device)
    prior_scale = torch.ones(prior_model.loc.shape[1]).to(device) * F.softplus(torch.tensor(-2.).to(device), beta=1, threshold=20) / 6
    
    prior_lpe_loc = torch.zeros(prior_model.lpe_loc.shape[1:]).to(device)
    prior_lpe_scale = torch.ones(prior_model.lpe_loc.shape[1:]).to(device) * F.softplus(torch.tensor(-2.).to(device), beta=1, threshold=20) / 6

    if config['patch']:
        prior_h_loc = torch.zeros(prior_model.h_loc.shape[-1]).to(device)
        prior_h_scale = torch.ones(prior_model.h_loc.shape[-1]).to(device) * F.softplus(torch.tensor(-2.).to(device), beta=1, threshold=20) / 6
        
        prior_hh_loc = torch.zeros(prior_model.hh_loc.shape[-1]).to(device)
        prior_hh_scale = torch.ones(prior_model.hh_loc.shape[-1]).to(device) * F.softplus(torch.tensor(-2.).to(device), beta=1, threshold=20) / 6
    else:
        prior_h_loc = None
        prior_h_scale = None

        prior_hh_loc = None
        prior_hh_scale = None

    # some training initial settings
    n_epoch = 200  # the first SGD epoch number
    n_em_iter = 550  # the total coordinate ascent iter number

    # coordinate descent loop
    for iter in tqdm(range(n_em_iter)):
        # train q and mappings
        prior_model.train(n_epoch,
                          2e-4,
                          X,
                          Y,
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
                          training_mappings=True, 
                          verbose=False)
        n_epoch = 100  # after the first iteration, change epoch to 100

        # adjust kl beta
        with torch.no_grad():
            kls = prior_model.calculate_kl(prior_loc,
                                           prior_scale,
                                           prior_lpe_loc,
                                           prior_lpe_scale,
                                           prior_h_loc,
                                           prior_h_scale,
                                           prior_hh_loc,
                                           prior_hh_scale).item()
        kls = (kls / np.log(2.)) / X.shape[0] # calculate average KL in bit
        # adjust beta according to kl
        if kls > budget_max:
            kl_beta *= 1.5
        if kls < budget_min:
            kl_beta /= 1.5
        # clamp KL within a reasonable range
        if kl_beta > 1:
            kl_beta = 1
        if kl_beta < 1e-20:
            kl_beta = 1e-20

        # update prior
        prior_loc = prior_model.loc.clone().detach().mean(0)
        prior_scale = (prior_model.st(prior_model.log_scale.clone().detach()) ** 2).mean(0) + prior_model.loc.clone().detach().var(0)
        prior_scale = prior_scale ** 0.5

        prior_lpe_loc = prior_model.lpe_loc.clone().detach().mean([0])
        prior_lpe_scale = (prior_model.st(prior_model.lpe_log_scale.clone().detach()) ** 2).mean([0]) + prior_model.lpe_loc.clone().detach().var([0])
        prior_lpe_scale = prior_lpe_scale ** 0.5

        if config['patch']:
            prior_h_loc = prior_model.h_loc.clone().detach().mean([0])
            prior_h_scale = (prior_model.st(prior_model.h_log_scale.clone().detach()) ** 2).mean([0]) + prior_model.h_loc.clone().detach().var([0])
            prior_h_scale = prior_h_scale ** 0.5

            prior_hh_loc = prior_model.hh_loc.clone().detach().mean([0])
            prior_hh_scale = (prior_model.st(prior_model.hh_log_scale.clone().detach()) ** 2).mean([0]) + prior_model.hh_loc.clone().detach().var([0])
            prior_hh_scale = prior_hh_scale ** 0.5

        # every 10 steps: print training psnr/rmsd and save checkpoints
        if (iter) % 10 == 0 or iter == n_em_iter - 1:
            with torch.no_grad():
                y_hat = prior_model.forward(X, linear_transform, upsample_net, False)
                if args.dataset not in ['protein']:
                    mses = ((y_hat - Y) ** 2).reshape(train_size, -1).mean(1).cpu().numpy() if config['patch']==False else ((y_hat - Y) ** 2).mean().cpu().numpy() 
                    print("Training PSNR %.4f" % (20 * np.mean(np.log10(1 / mses ** 0.5))) + "; Training KL %.4f" % kls, flush=True)
                else:
                    mses = ((y_hat - Y) ** 2).reshape(train_size, -1).mean(1).cpu().numpy() if config['patch']==False else ((y_hat - Y) ** 2).mean().cpu().numpy() 
                    mses = mses * 3 # note that for rmsd, the mse of xyz should be summed instead of averaged up.
                    print("Training RMSD %.4f" % (np.mean(mses**0.5)*25) + "; Training KL %.4f" % kls, flush=True) # do not forget to scale back by 25
            
                # save checkpoints
                # get average log_scale of all training instances
                average_training_log_scale = prior_model.log_scale.clone().detach().mean(0)
                average_training_lpe_log_scale = prior_model.lpe_log_scale.clone().detach().mean([0]).flatten()
                if config['patch']:
                    average_training_h_log_scale = prior_model.h_log_scale.clone().detach().mean([0]).flatten()
                    average_training_hh_log_scale = prior_model.hh_log_scale.clone().detach().mean([0]).flatten()
                else:
                    average_training_h_log_scale = None
                    average_training_hh_log_scale = None

                # get grouping by training set's average kl
                q_loc = torch.cat([prior_model.loc.flatten(start_dim=1),
                                   prior_model.lpe_loc.flatten(start_dim=1)], -1
                                  )
                q_scale = torch.cat([prior_model.st(prior_model.log_scale).flatten(start_dim=1),
                                     prior_model.st(prior_model.lpe_log_scale).flatten(start_dim=1)], -1
                                    )
                p_loc = torch.cat([prior_loc.flatten(),
                                   prior_lpe_loc.flatten()]
                                  )
                p_scale = torch.cat([prior_scale.flatten(),
                                     prior_lpe_scale.flatten()]
                                    )

                group_idx, group_start_index, group_end_index, group2param, param2group, n_groups, group_kls, weights = get_grouping(
                    q_loc, q_scale, p_loc, p_scale)
                
                if config['patch']:
                    h_p_loc = torch.cat([prior_h_loc])
                    h_p_scale = torch.cat([prior_h_scale])
                    h_q_loc = torch.cat([prior_model.h_loc], -1)
                    h_q_scale = torch.cat([prior_model.st(prior_model.h_log_scale)], -1)

                    h_group_idx, \
                    h_group_start_index, \
                    h_group_end_index, \
                    h_group2param, \
                    h_param2group, \
                    h_n_groups, \
                    h_group_kls, \
                    h_weights = get_grouping(
                        h_q_loc, 
                        h_q_scale, 
                        h_p_loc, 
                        h_p_scale)
                    
                    hh_p_loc = torch.cat([prior_hh_loc])
                    hh_p_scale = torch.cat([prior_hh_scale])
                    hh_q_loc = torch.cat([prior_model.hh_loc], -1)
                    hh_q_scale = torch.cat([prior_model.st(prior_model.hh_log_scale)], -1)
                    hh_group_idx, \
                    hh_group_start_index, \
                    hh_group_end_index, \
                    hh_group2param, \
                    hh_param2group, \
                    hh_n_groups, \
                    hh_group_kls, \
                    hh_weights = get_grouping(
                        hh_q_loc, 
                        hh_q_scale, 
                        hh_p_loc, 
                        hh_p_scale)
                else:
                    h_group_idx = None
                    h_group_start_index = None
                    h_group_end_index = None
                    h_group2param = None
                    h_param2group = None
                    h_n_groups = None
                    h_group_kls = None
                    h_weights = None
                    hh_group_idx = None
                    hh_group_start_index = None
                    hh_group_end_index = None
                    hh_group2param = None
                    hh_param2group = None
                    hh_n_groups = None
                    hh_group_kls = None
                    hh_weights = None



            # save
            file_name = "CONV_PRIOR_train_size_%d" % train_size + "_max_bitrate=%.3f.pkl" % args.max_bitrate
            with open(args.saving_dir + file_name, "wb") as f:
                pickle.dump(
                    (group_idx,
                     group_start_index,
                     group_end_index,
                     group2param,
                     param2group,
                     n_groups,
                     group_kls,
                     weights),
                    f)
                pickle.dump(
                    (p_loc,
                     p_scale,
                     kl_beta,
                     torch.cat([average_training_log_scale, average_training_lpe_log_scale])),
                    f)
                pickle.dump(
                    (h_group_idx,
                     h_group_start_index,
                     h_group_end_index,
                     h_group2param,
                     h_param2group,
                     h_n_groups,
                     h_group_kls,
                     h_weights),
                    f)
                pickle.dump(
                    (h_p_loc,
                     h_p_scale,
                     kl_beta,
                     torch.cat([average_training_h_log_scale])),
                    f)
                pickle.dump(
                    (hh_group_idx,
                     hh_group_start_index,
                     hh_group_end_index,
                     hh_group2param,
                     hh_param2group,
                     hh_n_groups,
                     hh_group_kls,
                     hh_weights),
                    f)
                pickle.dump(
                    (hh_p_loc,
                     hh_p_scale,
                     kl_beta,
                     torch.cat([average_training_hh_log_scale])),
                    f)
                pickle.dump(linear_transform, f)
                pickle.dump(upsample_net, f)


if __name__ == '__main__':
    main()
