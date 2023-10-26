
# Dataset Preparation



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
