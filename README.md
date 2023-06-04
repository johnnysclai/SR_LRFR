# SR LRFR
PyTorch implementation of [Low-Resolution Face Recognition Based on Identity-Preserved Face Hallucination](https://ieeexplore.ieee.org/document/8803782).

### Citation
If you find this work useful for your research, please consider cite our paper:
```
@inproceedings{lai2019low,
  title={Low-Resolution Face Recognition Based on Identity-Preserved Face Hallucination},
  author={Lai, Shun-Cheung and He, Chen-Hang and Lam, Kin-Man},
  journal={IEEE International Conference on Image Processing},
  pages={1173-1177},
  year={2019},
  month={Sep.}
}
```

### Requirements
- Python 3 ([Anaconda](https://www.anaconda.com) installation is recommended)
- numpy
- [PyTorch >= 0.4.1](https://pytorch.org)
- torchvision
- OpenCV 3
- tqdm (progress bar): ``pip install tqdm``

Tested environment: Ubuntu 16.04 with Python 3.6.5, OpenCV 3.4.1, PyTorch 0.4.1 (CUDA 9.2 & cuDNN 7.1)

### Low-resolution face verification experiment
Clone this repository
```
git clone https://github.com/johnnysclai/SR_LRFR
cd SR_LRFR/
```

To conduct low-resolution face verification, please download and extract the LFW database and 6,000 pairs file from [here](http://vis-www.cs.umass.edu/lfw/#download). Or, you just run the following commands:

```
mkdir datasets
cd datasets/
wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
tar -xvzf lfw.tgz
cd ../data
wget http://vis-www.cs.umass.edu/lfw/pairs.txt
cd ..
```
Now, you have the LFW database in ``datasets/lfw/`` and the 6,000 pairs file ``data/pairs.txt``. We have used [MTCNN](https://github.com/TropComplique/mtcnn-pytorch) to detect five facial landmarks, which are saved in ``data/LFW.csv``.

Extract the pre-trained checkpoints (models are compressed into parts as they are too large):
```
cd pretrained/
# Extract edsr_baseline.pth and edsr_lambda0.5.pth
7z x edsr_baseline.7z.001
7z x edsr_lambda0.5.7z.001
cd ..
```

Run the following commands to obtain the face verification results from our pre-trained models:
```
cd src/
bash lfw_verification.sh
```
You should be able to get the following results:

| FNet                                          | Method/SRNet                     | 7x6   | 14x12 | 28x24 | 112x96 |
| --------------------------------------------- | -------------------------------- | ----- | ----- | ----- | ------ |
| [SphereFace-20 (SFace)](pretrained/sface.pth) | Bicubic                          | 59.03 | 82.75 | 97.60 | 99.07  |
|                                               | [EDSR (lambda=0)](pretrained/)   | 73.42 | 92.25 | 98.48 | -      |
|                                               | [EDSR (lambda=0.5)](pretrained/) | 84.03 | 94.73 | 98.85 | -      |

*Note that [SphereFace-20 (SFace) model](pretrained/sface.pth) is converted from the [official released model](https://github.com/wy1iu/sphereface#models) using [extract-caffe-params](https://github.com/nilboy/extract-caffe-params).

### Training
Please download the CelebA dataset from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8) and extract it to ```../../../Datasets/```.

Train with single GPU: ```python train.py --gpu_ids=0```

Train with two GPU: ```python train.py --gpu_ids=0,1```

Final model will be saved in ```./checkpoint/master/```

## References
- LFW: [project page](http://vis-www.cs.umass.edu/lfw/) and [paper](http://vis-www.cs.umass.edu/lfw/lfw.pdf)
- MTCNN: [project page](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html), [paper](https://ieeexplore.ieee.org/abstract/document/7553523) and [code](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- SphereFace: [paper](https://arxiv.org/abs/1704.08063) and [code](https://github.com/wy1iu/sphereface)
- EDSR: [paper](https://ieeexplore.ieee.org/document/8014885) and [code](https://github.com/thstkdgus35/EDSR-PyTorch) 
- extract-caffe-params: https://github.com/nilboy/extract-caffe-params
