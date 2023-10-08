from prior_model import *

import torch
import argparse
import os
import pickle
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--train_size', type=int, default=None)
    parser.add_argument("--dataset", default="cifar", choices=("cifar", "kodak",), )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_bitrate", type=float, required=True)
    parser.add_argument("--saving_dir", default="./")
    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_args()
    in_dim = 32
    hidden_dims = [32, ] * 3
    out_dim = 3
    batch_size = None

    training_images = sorted(os.listdir(args.train_dir))
    
    training_images_path = []
    for img_name in tqdm(training_images):
        if args.train_dir[-1] == "/":
            image_path = args.train_dir + img_name
        else:
            image_path = args.train_dir + "/" + img_name
        training_images_path.append(image_path)

    entire_image_number = args.train_size if args.dataset == 'cifar' else (args.train_size // 96)
    np.random.seed(args.seed)
    idx = np.random.choice(len(training_images_path), entire_image_number, False)
    np.random.seed(None)
    training_images_path = [training_images_path[i] for i in idx]

    X, Y = load_dataset(training_images_path,
                        feature_size=16,
                        patch=True,
                        patch_size=64)
    X = X.to(args.device)
    Y = Y.to(args.device)

    train_size = X.shape[0]
    if batch_size == None:
        batch_size = train_size

    print("Prior is trained on %d patches/images." % train_size, flush=True)

    # defined model and mapping
    device = args.device

    prior_model = PriorBNNmodel(in_dim=in_dim,
                                hidden_dims=hidden_dims,
                                out_dim=out_dim,
                                training_set_size=train_size,
                                pixel_size=32 if args.dataset == 'cifar' else 64,
                                upsample_factor=16,
                                latent_dim=128,
                                ).to(device)
    mapping = Mapping(prior_model.dims).to(device)
    scale_mapping = Upsample().to(device)

    kl_beta = 1e-8
    budget_max = args.max_bitrate * 32 * 32 if args.dataset == "cifar" else args.max_bitrate * 64 * 64
    budget_min = max(0.1, (args.max_bitrate - 0.3)) * 32 * 32 if args.dataset == "cifar" else max(0.01, (args.max_bitrate - 0.05)) * 64 * 64
    assert budget_min <= budget_max

    # initialize priors
    prior_loc = torch.zeros(prior_model.loc.shape[1]).to(device)
    prior_scale = torch.ones(prior_model.loc.shape[1]).to(device) * F.softplus(torch.tensor(-2.).to(device),
                                                                                    beta=1,
                                                                                    threshold=20) / 6

    c_prior_loc = torch.zeros(prior_model.c_loc.shape[1:]).to(device)
    c_prior_scale = torch.ones(prior_model.c_loc.shape[1:]).to(device) * F.softplus(torch.tensor(-2.).to(device),
                                                                                    beta=1,
                                                                                    threshold=20) / 6

    h_loc_prior_loc = torch.zeros(prior_model.loc.shape[-1]).to(device)
    h_loc_prior_scale = torch.ones(prior_model.loc.shape[-1]).to(device) * F.softplus(torch.tensor(-2.).to(device), beta=1,
                                                                      threshold=20) / 6
    
    hh_loc_prior_loc = torch.zeros(prior_model.loc.shape[-1]).to(device)
    hh_loc_prior_scale = torch.ones(prior_model.loc.shape[-1]).to(device) * F.softplus(torch.tensor(-2.).to(device), beta=1,
                                                                      threshold=20) / 6


    # some training initial settings
    n_epoch = 200  # the first SGD epoch number
    n_em_iter = 550  # the total coordinate ascent iter number

    # coordinate descent loop
    for iter in tqdm(range(n_em_iter)):
        # train q and mappings
        prior_model.train(n_epoch,
                          batch_size,  # batch size
                          2e-4,  # lr
                          X,
                          Y,
                          prior_loc,
                          prior_scale,
                          c_prior_loc,
                          c_prior_scale,
                          h_loc_prior_loc,
                          h_loc_prior_scale,
                          hh_loc_prior_loc,
                          hh_loc_prior_scale,
                          mapping,
                          scale_mapping,
                          training_mapping=True,  # train mapping in the first 100 iterations
                          kl_beta=kl_beta)
        n_epoch = 100  # after the first iteration, change epoch to 100

        # adjust kl beta
        with torch.no_grad():
            kls = prior_model.calculate_kl(prior_loc,
                                           prior_scale,
                                           c_prior_loc,
                                           c_prior_scale,
                                           h_loc_prior_loc,
                                           h_loc_prior_scale,
                                           hh_loc_prior_loc,
                                           hh_loc_prior_scale,
                                           ).item()
        kls = (kls / np.log(2.)) / X.shape[0]
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
        prior_scale = (prior_model.st(prior_model.log_scale.clone().detach()) ** 2).mean(
            0) + prior_model.loc.clone().detach().var(0)
        prior_scale = prior_scale ** 0.5

        c_prior_loc = prior_model.c_loc.clone().detach().mean([0])
        c_prior_scale = (prior_model.st(prior_model.c_log_scale.clone().detach()) ** 2).mean([0]) + prior_model.c_loc.clone().detach().var([0])
        c_prior_scale = c_prior_scale ** 0.5

        h_loc_prior_loc = prior_model.hyper_loc_loc.clone().detach().mean([0])
        h_loc_prior_scale = ((prior_model.st(prior_model.hyper_loc_log_scale.clone().detach()) ** 2).mean([0]) + prior_model.hyper_loc_loc.clone().detach().var([0]))**0.5

        hh_loc_prior_loc = prior_model.hh_loc_loc.clone().detach().mean([0])
        hh_loc_prior_scale = ((prior_model.st(prior_model.hh_loc_log_scale.clone().detach()) ** 2).mean([0]) + prior_model.hh_loc_loc.clone().detach().var([0]))**0.5


        # check training psnr
        if (iter) % 10 == 0 or iter == n_em_iter - 1:
            with torch.no_grad():
                y_hat = prior_model.forward(X, mapping, scale_mapping)
                mse = ((y_hat - Y) ** 2).mean((-1, -2)).cpu().numpy() if args.dataset == 'cifar' else ((y_hat - Y) ** 2).mean().cpu().numpy() 
                print("Training PSNR %.4f" % (20 * np.mean(np.log10(1 / mse ** 0.5))) + "; Training KL %.4f" % kls,
                      flush=True)

                # get average log_scale of all training instances
                average_training_log_scale = prior_model.log_scale.clone().detach().mean(0)
                average_training_c_log_scale = prior_model.c_log_scale.clone().detach().mean([0]).flatten()
                average_training_h_loc_log_scale = prior_model.hyper_loc_log_scale.clone().detach().mean([0]).flatten()
                average_training_hh_loc_log_scale = prior_model.hh_loc_log_scale.clone().detach().mean([0]).flatten()


                # get grouping by training set
                q_loc = torch.cat([prior_model.loc.flatten(start_dim=1),
                                   prior_model.c_loc.flatten(start_dim=1)], -1
                                  )
                q_scale = torch.cat([prior_model.st(prior_model.log_scale).flatten(start_dim=1),
                                     prior_model.st(prior_model.c_log_scale).flatten(start_dim=1)], -1
                                    )
                p_loc = torch.cat([prior_loc.flatten(),
                                   c_prior_loc.flatten()]
                                  )
                p_scale = torch.cat([prior_scale.flatten(),
                                     c_prior_scale.flatten()]
                                    )

                group_idx, group_start_index, group_end_index, group2param, param2group, n_groups, group_kls, weights = get_grouping(
                    q_loc, q_scale, p_loc, p_scale)

                h_p_loc = torch.cat([h_loc_prior_loc])
                h_p_scale = torch.cat([h_loc_prior_scale])
                h_q_loc = torch.cat([prior_model.hyper_loc_loc], -1
                                   )
                h_q_scale = torch.cat([prior_model.st(prior_model.hyper_loc_log_scale)], -1)
                h_group_idx, h_group_start_index, h_group_end_index, h_group2param, h_param2group, h_n_groups, h_group_kls, h_weights = get_grouping(
                    h_q_loc, h_q_scale, h_p_loc, h_p_scale)
                
                hh_p_loc = torch.cat([hh_loc_prior_loc])
                hh_p_scale = torch.cat([hh_loc_prior_scale])
                hh_q_loc = torch.cat([prior_model.hh_loc_loc], -1
                                   )
                hh_q_scale = torch.cat([prior_model.st(prior_model.hh_loc_log_scale)], -1)
                hh_group_idx, hh_group_start_index, hh_group_end_index, hh_group2param, hh_param2group, hh_n_groups, hh_group_kls, hh_weights = get_grouping(
                    hh_q_loc, hh_q_scale, hh_p_loc, hh_p_scale)


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
                     torch.cat([average_training_log_scale, average_training_c_log_scale])),
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
                     torch.cat([average_training_h_loc_log_scale])),
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
                     torch.cat([average_training_hh_loc_log_scale])),
                    f)
                pickle.dump(mapping, f)
                pickle.dump(scale_mapping, f)


if __name__ == '__main__':
    main()