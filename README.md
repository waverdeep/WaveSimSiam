# WaveSimSiam

Self-Supervised Learning for General Audio Representation of Raw Waveform

## Description

자기지도학습은 데이터가 내포하고 있는 일반적인 표현을 학습하기 위한 방법론이다. 
본 프로젝트에서는 SimSiam 모델을 기반으로 원시 오디오 파형에서 일반적인 표현을 학습할 수 있는 WaveSimSiam모델을 제안한다. 
WaveSimSiam은 오디오의 일반적인 표현을 학습할 수 있도록 원시 오디오 파형 데이터에 적합한 Augmentation Layer와 Encoding Layer를 설계하였다. 
제안한 모델을 평가하기 위해 선형 분류기 평가 방식으로 분류와 인식 기반의 실제 작업을 진행하였다.
WaveSimSiam은 대부분의 실제 작업에서 기존 논문들보다 향상된 성능결과를 보였다. 

## Getting Started

### Datasets
* [FSD50K](https://arxiv.org/abs/2010.00475) - Training Pretext task
* [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) - Downstream task
* [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) - Downstream task
* [SpeechCommandV2](https://arxiv.org/abs/1804.03209) - Downstream task
* [RAVDESS](https://smartlaboratory.org/ravdess/) - Downstream task
* [VoxForge](http://www.voxforge.org/home) - Downsteam task

### Dependencies

* Linux Ubuntu, Nvidia Docker, Python
* adamp 0.3.0
* scikit-learn 1.0.2
* numpy 1.21.6
* tensorboard 2.8.0
* torch 1.12.0
* torchvision 0.13.0
* torchaudio 0.12.0
* tqdm 4.63.1
* sox 1.4.1
* soundfile 0.10.3
* natsort 8.1.0
* [WavAugment](https://github.com/facebookresearch/WavAugment)

### Pretext Task

1. Make up own your configuration file.  (There is an pretext example in the config folder)
2. You can modify this part at [train.py](https://github.com/waverDeep/ImageBYOL/blob/master/train.py)
```
...
...

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # input tranning cuda device number


...
...


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveSimSiam')
    parser.add_argument("--configuration", required=False,
                        default='./config/write down your configuration file name')
                        
...
...

```
3. And then, start pretext task training!
```
python train.py
```


### Downstream Task

Currently, only transfer learning is implemented in this project.
1. Make up own your configuration file.  (There is an transfer learning example in the config folder)
2. You can modify this part at [train.py](https://github.com/waverDeep/ImageBYOL/blob/master/train.py)
```
...
...

os.environ['CUDA_VISIBLE_DEVICES'] = '1' # input tranning cuda device number


...
...


def main():
    parser = argparse.ArgumentParser(description='waverdeep - WaveSimSiam')
    parser.add_argument("--configuration", required=False,
                        default='./config/write down your configuration file name')
                        
...
...
```

3. And then, start pretext task training!

```
python train.py
```


## Authors

[waverDeep](https://github.com/waverDeep)

## Version History

    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

* [BYOL](https://github.com/lucidrains/byol-pytorch)
* [BYOL-A](https://github.com/nttcslab/byol-a)
* [Spijkervet/contrastive-predictive-coding](https://github.com/Spijkervet/contrastive-predictive-coding)
* [jefflai108/Contrastive-Predictive-Coding-PyTorch](https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch)
* [WavAugment](https://github.com/facebookresearch/WavAugment)
