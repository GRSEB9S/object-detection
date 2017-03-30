## Fast Multispectral Deep Fusion Networks for Human Detection

### Table of Contents
1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Results](#results)
5. [Installation](#installation)

------
### Introduction


Current state-of-the-art object detection algorithms produce very nice results for images with good light condition and almost useless during night time. The main idea of this work utilize information from thermal infrared sensors and RGB camera to avoid such illumination problems.

Also, it is hard to launch complex deep neural networks on embedded deivices with near real-time performance, 

------
### Datasets

#### Piropo

#### KAIST
#### CVC-14


### Results

| Base Network  | GPU Titan X (FPS)  | CPU I7-5820K (FPS)  | Raspberry PI Model 3B (single image detectionin seconds)| Model Size (MB)|
| ------------- | :----------: | :-------------------: | :------------: | :--:  |
| CaffeNet      | 80           |      3.5              |    2.8564      | 26.1  |
| SqueezeNet    | 85           |      4.2              |    1.9142      |       |
| Resnet-18     | 74           |      0.1              |    7.5         | 58.6  |
| VGG16_reduced | --           |                       |    54.76       | 104.3 |



*all tests were performed for [this](https://www.youtube.com/watch?v=h0qhZK0eGZY) video



|Base Network                       | Person | Car  | Bus | Bicycle | Motorbike | Train | Aeroplane | AP |
| --------------------------------- | :----: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
|CaffeNet                           | 48.06 | 54.62 | 56.49 | 53.64 | 54.93 | 57.12 | 47.83 | 41.40 |
|SqueezeNet                         | 55.11 | 55.52 | 65.91 | 58.21 | 61.40 | 68.60 | 55.46 | 49.55 |
|Resnet-18                          | 72.81 | 79.32 | 73.47 | 76.31 | 75.81 | 75.40 | 65.22 | 65.15 |
|VGG16_reduced-mxnet original model | 74.39 | 81.77 | 77.91 | 79.69 | 77.06 | 84.01 | 72.15 | 71.57 |

*you can download weights via this [link](https://goo.gl/Uwyom7) 

------


