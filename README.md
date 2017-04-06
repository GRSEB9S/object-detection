## Fast Multispectral Deep Fusion Networks for Human Detection

### TO-DO
0. Add fusion model description 
------

### Table of Contents
1. [Detection](#detection)
2. [References](#references)

------

### Detection 

#### Fig 1. Validation overall accuracy (including background prediction)
![alt tag](https://github.com/osin-vladimir/ms-thesis-skoltech/blob/master/notebooks/img/overall%20accuracy.png)

#### Fig 2. Validation object accuracy (only foreground object predictions)
![alt tag](https://github.com/osin-vladimir/ms-thesis-skoltech/blob/master/notebooks/img/object%20accuracy.png)

#### Fig 3. Validation smooth L1 loss 
![alt tag](https://github.com/osin-vladimir/ms-thesis-skoltech/blob/master/notebooks/img/smooth%20l1%20loss.png)

#### Table 1. Performance results 
| Base Network  | GPU Titan X (FPS)  | CPU I7-5820K | Raspberry PI 3 | Model Size (MB)|
| ------------- | :----------: | :-------------------: | :---------: | :--:  |
| CaffeNet      | 102          |      0.39             |    2.6      | 26.1  |
| SqueezeNet    | 138          |      0.19             |    1.7      | 17.8  |
| Resnet-18     | 79           |      0.71             |    7.3      | 58.6  |
| VGG16_reduced | 45           |      3.2              |    55.6     | 104.3 |

*demo video with fps test of resnet-18 [here](https://www.youtube.com/watch?v=QvC_bejEtzY) 

#### Table 2. Detection results for various base networks
|Base Network                       | Person | Car  | Bus | Bicycle | Motorbike | Train | Aeroplane | mAP |
| --------------------------------- | :----: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
|CaffeNet                           | 43.92 | 51.11 | 51.94 | 52.31 | 55.56 | 60.44 | 49.37 | 40.56 |
|SqueezeNet                         | 57.02 | 56.67 | 66.09 | 62.27 | 64.73 | 68.42 | 56.71 | 51.68 |
|Resnet-18                          | 72.23 | 79.07 | 74.98 | 77.89 | 79.43 | 79.24 | 70.98 | 67.15 |
|VGG16 (main base network from SSD paper) | 74.39 | 81.77 | 77.91 | 79.69 | 77.06 | 84.01 | 72.15 | 71.57 |

*you can download weights via this [link](https://goo.gl/Uwyom7) 

------
#### Datasets
For Single Shot Multibox Detector training/testing procedures I used PASCAL VOC 07/12 datasets.
For training/testing procedures of multispectral model I used reasonable subsets of Caltech Pedestrian Detector and KAIST Multispectral Dataset.

Links: [pascal](http://host.robots.ox.ac.uk:8080/pascal/VOC/voc2007/index.html), [caltech](https://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/), [kaist](https://sites.google.com/site/pedestrianbenchmark/)

------
### References 
1. [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)
2. [Multispectral Pedestrian Detection: Benchmark Dataset and Baseline](https://goo.gl/ZF9v6r)
3. [Pedestrian Detection:An Evaluation of the State of the Art](https://goo.gl/oMjBFq)

------
### Github repositories
1. [SSD: Single Shot MultiBox Detector](https://github.com/weiliu89/caffe/tree/ssd)
2. [Joshua Z. Zhang implementation of SSD for MXnet](https://github.com/zhreshold/mxnet-ssd)



