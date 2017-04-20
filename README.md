## Fast Multispectral Deep Fusion Neural Networks


### Table of Contents
1. [Installation](#installation)
2. [Detection](#detection)
2. [References](#references)

------

### Installation 

- download mxnet src from [here]()
- install mxnet for python with all required dependencies

```
cd ~/mxnet/setup-utils
bash install-mxnet-ubuntu-python.sh
```

- create symbolic link to the data folder for your dataset

```
ln -s /path/to/pascal /path/to/ssd/data/pascal
```

------

### Detection 

#### Fig 1. Validation overall accuracy (including background prediction)
![alt tag](https://github.com/osin-vladimir/ms-thesis-skoltech/blob/master/notebooks/img/overall%20accuracy.png)

#### Fig 2. Validation object accuracy (only foreground object predictions)
![alt tag](https://github.com/osin-vladimir/ms-thesis-skoltech/blob/master/notebooks/img/object%20accuracy.png)

#### Fig 3. Validation smooth L1 loss 
![alt tag](https://github.com/osin-vladimir/ms-thesis-skoltech/blob/master/notebooks/img/smooth%20l1%20loss.png)

#### Table 1. Performance results 
| Base Network  | GPU Titan X (FPS)  | CPU I7-5820K | Raspberry PI 3 | Model Size (MB)| Total params|
| ------------- | :----------: | :-------------------: | :---------: | :--:  | :---: |
| CaffeNet      | 102          |      0.39             |    2.6      | 26.1  | 3737  |
| SqueezeNet    | 138          |      0.19             |    1.7      | 17.8  | 5080  |
| Resnet-18     | 79           |      0.71             |    7.3      | 58.6  | 7225  |
| VGG16_reduced | 45           |      3.2              |    55.6     | 104.3 | 8633  |

*demo video with fps test of resnet-18 [here](https://www.youtube.com/watch?v=QvC_bejEtzY) 

#### Table 2. Detection results for various base networks
|Base Network                       | Person | Car  | Bus | Bicycle | Motorbike | Train | Aeroplane | mAP |
| --------------------------------- | :----: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
|CaffeNet                           | 43.92 | 51.11 | 51.94 | 52.31 | 55.56 | 60.44 | 49.37 | 40.56 |
|SqueezeNet                         | 57.02 | 56.67 | 66.09 | 62.27 | 64.73 | 68.42 | 56.71 | 51.68 |
|Resnet-18                          | 72.23 | 79.07 | 74.98 | 77.89 | 79.43 | 79.24 | 70.98 | 67.15 |
|VGG16_reduced                      | 74.39 | 81.77 | 77.91 | 79.69 | 77.06 | 84.01 | 72.15 | 71.57 |

*you can download weights via this [link](https://goo.gl/Uwyom7) 

------
#### Datasets
Single Shot Multibox Detector train/test:

- PASCAL VOC 07/12 Dataset [link](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/index.html)

Multispectral Detectors (reasonable subsets of the following datasets):

- KAIST Multispectral Dataset [link](https://sites.google.com/site/pedestrianbenchmark/)
- CVC-14: Visible-FIR Day-Night Pedestrian Sequence Dataset [link](http://adas.cvc.uab.es/elektra/enigma-portfolio/cvc-14-visible-fir-day-night-pedestrian-sequence-dataset/)

------
### References 
1. [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
2. [Multispectral Pedestrian Detection: Benchmark Dataset and Baseline](https://goo.gl/ZF9v6r)

------
### Github repositories
1. [SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd)
2. [Joshua Z. Zhang implementation of SSD for MXnet](https://github.com/zhreshold/mxnet-ssd)



