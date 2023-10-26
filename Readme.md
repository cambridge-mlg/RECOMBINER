# RECOMBINER: Robust and Enhanced Compression with Bayesian Implicit Neural Representations

PyTorch implementation of the INR-based codec [RECOMBINER](https://arxiv.org/abs/2309.17182). 

## Installation

We suggest using the following commands.

```
conda create --name $ENV_NAME
conda activate $ENV_NAME
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r requirements.txt
```


## Dataset Preparation

### Kodak

Please put training images and test images into ```train_dir``` and ```test_dir``` respectively.
For the training set, we suggest using [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). 
To align with the resolution of Kodak, the training images should be randomly cropped into patches with resolution of 512x768 or 768x512.


### CIFAR-10

Please put training images and test images into ```train_dir``` and ```test_dir``` respectively.

### Video (UCF-101)

First put training and test clips into ```train_dir``` and ```test_dir```.
Then call ```dataset.process_video_datasets``` to process training and test datasets.
The processed video tensor will be saved in binary files in the specified directory.


### Audio (LibriSpeech)


### 3D Protein Structure


## Execution

### Training RECOMBINER

### Compression Test Data Points


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
