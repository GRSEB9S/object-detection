### Description 


### SSD for Indoor Human Detection - RGB

| Base Network  | FPS(Titan X) | Raspberry PI Model 3B | Model Size (MB)|
| ------------- | ------------ | --------------------- | -------------- | 
| CaffeNet      | 45           |                       |                |
| SqueezeNet    | 25           |                       |                |
| Resnet-18     | 45           |                       |                |

*all test were performed for [this](https://www.youtube.com/watch?v=h0qhZK0eGZY) video


##### Evaluations results for PASCAL VOC 2007 test set

|Base Network                       | Person | Car  | Bus | Bicycle | Motorbike | Train | Aeroplane |
| --------------------------------- | ----- | ----- | ---   | ------- | --------- | ----- | --------- | 
|CaffeNet                           | ----- | ----- | ---   |-------|---------|-----  |---------|      
|SqueezeNet                         | ----- | ----- | ---   |-------|---------|-----  |---------|       
|[Resnet-18](https://goo.gl/rVnQLx) | 74.39 | 81.09 | 77.22 | 76.72 | 79.12 | 77.73 | 70.18 | 60.72 |
|VGG16_red                          | ----- | ----- | ----- |-------|---------|-----  |---------|       

*you can download weights via link in network name

