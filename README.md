## Fast Multispectral Deep Fusion Networks for Human Detection

### Table of Contents
1. [Introduction](#introduction)
2. [Contribution](#contribution)
3. [Datasets](#datasets)
4. [Detection results](#detection_results)
5. [Installation](#installation)

------
### Introduction

Current state-of-the-art object detection algorithms produce very nice results for images with good light condition and almost useless during night time. The main idea of this work utilize information from thermal infrared sensors and RGB camera to avoid such illumination problems.
Also, it is hard to launch complex deep neural networks on embedded deivices with near real-time performance... 

------
### Contribution
------

### Datasets

#### PASCAL VOC


#### [KAIST Multispectral Pedestrian Dataset](https://sites.google.com/site/pedestrianbenchmark/)

KAIST developed imaging hardware consisting of a color camera, a thermal camera and a beam splitter to capture the aligned multispectral (RGB color + Thermal) images. With this hardware, they captured various regular traffic scenes at day and night time to consider changes in light conditions.The KAIST Multispectral Pedestrian Dataset consists of 95k color-thermal pairs (640x480, 20Hz) taken from a vehicle. All the pairs are manually annotated (person, people, cyclist) for the total of 103,128 dense annotations and 1,182 unique pedestrians. The annotation includes temporal correspondence between bounding boxes like Caltech Pedestrian Dataset. More infomation can be found in [CVPR 2015 paper](https://goo.gl/ZF9v6r).

### Detection 

#### Fig 1. Training/Validation curves



#### Table 1. Performance results 
| Base Network  | GPU Titan X (FPS)  | CPU I7-5820K | Raspberry PI 3 | Model Size (MB)|
| ------------- | :----------: | :-------------------: | :---------: | :--:  |
| CaffeNet      | ?            |      ?                |    ?        | 26.1  |
| SqueezeNet    | **140**      |      0.26             |    ?        | 17.8  |
| Resnet-18     | 79           |      0.71             |    7.5      | 58.6  |
| VGG16_reduced | 47           |      3.2              |    54.8     | 104.3 |

*demo video with fps test of resnet-18 [here](https://www.youtube.com/watch?v=h0qhZK0eGZY) 


#### Table 2. Detection results for various base networks
|Base Network                       | Person | Car  | Bus | Bicycle | Motorbike | Train | Aeroplane | AP |
| --------------------------------- | :----: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
|CaffeNet                           | 49.98 | 60.57 | 54.49 | 58.56 | 62.10 | 58.27 | 52.83 | 44.59 |
|SqueezeNet                         | 57.02 | 56.67 | 66.09 | 62.27 | 64.73 | 68.42 | 56.71 | 51.68 |
|Resnet-18                          | **72.23** | 79.07 | 74.98 | 77.89 | 79.43 | 79.24 | 70.98 | 67.15 |
|VGG16_reduced-mxnet **original model** | 74.39 | 81.77 | 77.91 | 79.69 | 77.06 | 84.01 | 72.15 | 71.57 |

*you can download weights via this [link](https://goo.gl/Uwyom7) 

------

### Thermal SSD

#### FPPI graph 
#### miss rate

------
### Fusion results (in development)

| Base Network  | Sum Fusion   | Convolution Fusion    |
| ------------- | :----------: | :-------------------: | 
| Resnet-18     |     --       |      --               | 

------

### Installation 
...
