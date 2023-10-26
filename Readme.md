# RECOMBINER: Robust and Enhanced Compression with Bayesian Implicit Neural Representations

Official PyTorch implementation of the INR-based codec [RECOMBINER](https://arxiv.org/abs/2309.17182). It sets a new SOTA on CIFAR-10 at low bitrates and achieves strong performance on other modalities comparing to other INR-based codecs. This repo provides implementations of RECOMBINER across modalities, including image (Kodak, CIFAR-10), audio, video, and protein 3D structure. 




## ⚙️ Installation

We suggest using the following commands.

```
conda create --name $ENV_NAME
conda activate $ENV_NAME
conda install pytorch=2.1.0 torchvision=0.16.0 torchaudio=2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip3 install -r requirements.txt
```


## 📁 Dataset Preparation


### Kodak

Please put training images and test images into ```train_dir``` and ```test_dir``` respectively.
For the training set, we suggest using [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). 
To align with the resolution of Kodak, the training images should be randomly cropped into patches with resolution of 512x768 or 768x512.


### CIFAR-10

Please put training images and test images into ```train_dir``` and ```test_dir``` respectively.

### Video (UCF-101)

First put training and test clips into ```train_dir``` and ```test_dir```.
Then call ```data.video.process_video_datasets``` to process training and test datasets.
The processed video tensor will be saved in binary files in the specified directory.


### Audio (LibriSpeech)

First put the test clips into ```LibriSpeech/test-clean```. Then call ```data.audio.process_audio_datasets``` to process training and test datasets.
This function will download LibriSpeech training set automatically and process the training & test sets in binary files in the specified directory.


We provide the 24 test data instances we used in the paper in ```test-clean.zip```. If you would like to compress other test set, it is important to follow the same structure to be compatible with the ```torchaudio.datasets.LIBRISPEECH``` API, i.e., 
```
|-- LibriSpeech
      |-- test-clean
            |-- speaker_id
                  |-- audio_id
                        |-- clip01.flac
                        |-- clip02.flac
                        ...
                  ...
            ...
```


### 3D Protein Structure


## 💻 Execution

### Training RECOMBINER

### Compression Test Data Points


## 🌟 Citation
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
