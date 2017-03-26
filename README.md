### SSD for Indoor Human Detection - RGB

| Base Network  | FPS(Titan X) | Raspberry PI Model 3B | Model Size (MB)|
| ------------- | ------------ | --------------------- | -------------- | 
| CaffeNet      | 45           |                       |                |
| SqueezeNet    | 25           |                       |                |
| Resnet-18     | 45           |                       |                |

*all tests were performed for [this](https://www.youtube.com/watch?v=h0qhZK0eGZY) video


##### Evaluations results for PASCAL VOC 2007 test set

|Base Network                       | Person | Car  | Bus | Bicycle | Motorbike | Train | Aeroplane |
| --------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ------| 
|CaffeNet                           | 49.36 | 57.81 | 57.39 | 59.25 | 60.43 | 62.19 | 52.62 |      
|SqueezeNet                         | 62.33 | 69.94 | 64.60 | 68.02 | 66.85 | 69.63 | 63.02 |       
|Resnet-18                          | 74.41 | 81.09 | 77.22 | 76.72 | 79.12 | 77.73 | 70.18 | 
|VGG16_red                          | ----- | ----- | ----- |-------|---------|-----  |---------|       

*you can download weights via this [link](https://goo.gl/Uwyom7) 

