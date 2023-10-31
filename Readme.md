# RECOMBINER: Robust and Enhanced Compression with Bayesian Implicit Neural Representations

Official PyTorch implementation of the INR-based codec [RECOMBINER](https://arxiv.org/abs/2309.17182). It sets a new SOTA on CIFAR-10 at low bitrates and achieves strong performance on other modalities comparing to other INR-based codecs. This repo provides implementations of RECOMBINER across modalities, including image (Kodak, CIFAR-10), audio, video, and protein 3D structure. 




## Installation

We suggest using the following commands.

```
conda create --name $ENV_NAME
conda activate $ENV_NAME
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r requirements.txt
```


## Dataset Preparation

Please refer to ```./data/Readme.md```.


## Execution

- **Training RECOMBINER**


 
```
python main_prior_training.py [--seed] [--train_dir] [--train_size] [--dataset] [--device] [--max_bitrate] [--saving_dir]
```

- **Compression Test Data Points**

```
python main_compression.py [--seed] [--test_dir] [--test_idx] [--dataset] [--device] [--prior_path] [--save_dir]
```

## Hyperparameters
You can also adjust the hyperparameters by modifying ```config.py```

## Citation
Please consider citing the following paper if you use this repo.
```
@misc{he2023recombiner,
      title={RECOMBINER: Robust and Enhanced Compression with Bayesian Implicit Neural Representations}, 
      author={Jiajun He and Gergely Flamich and Zongyu Guo and José Miguel Hernández-Lobato},
      year={2023},
      eprint={2309.17182},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
