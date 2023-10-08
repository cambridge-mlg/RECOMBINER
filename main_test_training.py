import numpy as np

from test_model import *
from prior_model import get_grouping_by_kl_with_fixed_order, Upsample
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--test_idx', type=int, required=True)  # if cifar, test 100 images at once. if kodak, test 1 at once
    parser.add_argument('--mixing', type=int, required=True) # mix if kodak
    parser.add_argument("--dataset", default="cifar", choices=("cifar", "kodak",), )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prior_path", required=True)
    args = parser.parse_args()
    return args

def get_test_paths(args):
    images = sorted(os.listdir(args.test_dir))
    data_paths = []
    for img_name in images:
        if args.test_dir[-1] == "/":
            image_path = args.test_dir + img_name
        else:
            image_path = args.test_dir + "/" + img_name
        data_paths.append(image_path)
    if args.dataset == 'cifar':
        test_start_idx = args.test_idx * 500
        test_end_idx = args.test_idx * 500 + 500
    else:
        test_start_idx = args.test_idx * 1
        test_end_idx = args.test_idx * 1 + 1
    return data_paths[test_start_idx: test_end_idx]

def main():
    # parse arguments
    args = parse_args()
    in_dim = 32
    hidden_dims = [32, ] * 3
    out_dim = 3
    device = args.device

    args.mixing = bool(args.mixing)
    if args.mixing:
        print("MIXING...", flush=True)
    else:
        print("NOT MIXING...", flush=True)

    # load priors
    with open(args.prior_path, "rb") as f:
        group_idx, group_start_index, group_end_index, group2param, param2group, n_groups, group_kls, weights = pickle.load(f)
        prior_loc, prior_scale, kl_beta, average_training_log_scale = pickle.load(f)
        h_group_idx, h_group_start_index, h_group_end_index, h_group2param, h_param2group, h_n_groups, h_group_kls, h_weights = pickle.load(f)
        h_prior_loc, h_prior_scale, _, h_average_training_log_scale = pickle.load(f)
        hh_group_idx, hh_group_start_index, hh_group_end_index, hh_group2param, hh_param2group, hh_n_groups, hh_group_kls, hh_weights = pickle.load(f)
        hh_prior_loc, hh_prior_scale, _, hh_average_training_log_scale = pickle.load(f)
        mapping = pickle.load(f)
        latent_mapping = pickle.load(f)

    # load and reorder priors
    # the variable starting with _ is the reordered one
    p_locs = prior_loc.clone()
    _p_locs = p_locs[param2group].to(device)
    p_log_scales = torch.log(torch.exp(prior_scale * 6) - 1).clone()
    _p_log_scales = p_log_scales[param2group].to(device)
    

    h_p_locs = h_prior_loc.clone()
    _h_p_locs = h_p_locs[h_param2group].to(device)
    h_p_log_scales = torch.log(torch.exp(h_prior_scale * 6) - 1).clone()
    _h_p_log_scales = h_p_log_scales[h_param2group].to(device)

    hh_p_locs = hh_prior_loc.clone()
    _hh_p_locs = hh_p_locs[hh_param2group].to(device)
    hh_p_log_scales = torch.log(torch.exp(hh_prior_scale * 6) - 1).clone()
    _hh_p_log_scales = hh_p_log_scales[hh_param2group].to(device)

    # initialize model
    test_image_paths = get_test_paths(args)
    if args.dataset == "kodak":
        print("Image path: " + test_image_paths[0])
    combnn = TestBNNmodel(in_dim=in_dim,
                          hidden_dims=hidden_dims,
                          out_dim=out_dim,
                          number_of_datapoints=len(test_image_paths) * (512*768//64//64 if args.dataset == "kodak" else 1),
                          upsample_factor=16,
                          latent_dim=128,
                          mix_datapoints=args.mixing,
                          pixel_size=64 if args.dataset == "kodak" else 32,
                          patch=True if args.dataset == "kodak" else False,
                          patch_size=64,
                          param_mapping=mapping,
                          coord_mapping=latent_mapping,
                          random_seed=args.seed,
                          w0=30.,
                          c=6.,
                          eps_beta_0=kl_beta,
                          eps_beta=0.05,
                          device="cuda",
                          n_pixels=32 * 32 if args.dataset != 'kodak' else 64**2,
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

                          search_space_size=32,
                          kl_buffer=0.2,
                          kl_adjust_epoch=0,
                          kl_adjust_gap=10,
                          kl_boundary=0.2).to(device)

    x, y = combnn._compress_prepare(image_paths=test_image_paths,
                                    feature_size=16,
                                    )
    x = x.to(device)
    y = y.to(device)
    _ = combnn._compress_train(x,
                               y,
                               n_epoch_kl=30000,
                               verbose=1,
                               lr=2e-4,
                               )
    combnn._compress_compress(x,
                              y,
                              n_epoch_compress=max(30000 // n_groups, 100),
                              verbose=1,
                              lr=2e-4,
                              fine_tune_gap=1,
                              compress_from_largest=True,
                              )

    if args.dataset == "kodak":
        psnrs = np.array([[PSNR(y.cpu().detach().numpy(), combnn.predict(x, random_seed=0).cpu().detach().numpy())]])
    else:
        psnrs = batch_PSNR(y.cpu().detach().numpy(), combnn.predict(x, random_seed=0).cpu().detach().numpy())

    # save
    if not os.path.exists('Prior_[' + args.prior_path + ("]_Mixing" if args.mixing else "_NotMixing")):
        os.makedirs('Prior_[' + args.prior_path + ("]_Mixing" if args.mixing else "_NotMixing"))
    saving_dir = 'Prior_[' + args.prior_path + ("]_Mixing" if args.mixing else "_NotMixing") + "/"

    file_name = "PSNR_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(saving_dir + file_name, psnrs, delimiter=",")

    file_name = "GroupIndex_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(saving_dir + file_name, combnn.compressed_groups_i, delimiter=",")

    file_name = "H_GroupIndex_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(saving_dir + file_name, combnn.h_compressed_groups_i, delimiter=",")

    file_name = "HH_GroupIndex_test_id_%d" % args.test_idx + ".csv"
    np.savetxt(saving_dir + file_name, combnn.hh_compressed_groups_i, delimiter=",")

if __name__ == '__main__':
    main()
