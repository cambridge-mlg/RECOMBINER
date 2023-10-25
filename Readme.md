# RECOMBINER: Robust and Enhanced Compression with Bayesian Implicit Neural Representations

PyTorch implementation of the INR-based codec [RECOMBINER](https://arxiv.org/abs/2309.17182). 

## Installation

We suggest using the following commands.

```
conda create --name $ENV_NAME
conda activate $ENV_NAME

```


## Dataset Preparation

### Kodak

Please put training images and test images into ```train_dir``` and ```test_dir``` respectively.
For the training set, we suggest using [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) cropped to the same size as the Kodim images.


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
If you use this repo, please cite the following paper.
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
