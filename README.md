


| Base Network  | GPU Titan X (FPS)  | CPU I7-5820K  | Raspberry PI Model 3B| Model Size (MB)|
| ------------- | :----------: | :-------------------: | :------------: | :--: |
| CaffeNet      | 80           |      3.5              |                | 26.1 |
| SqueezeNet    |              |                       |                |      |
| Resnet-18     | 74           |                       |                | 58.6 |
| VGG16_reduced | --           |                       |                | 104.3 |



*all tests were performed for [this](https://www.youtube.com/watch?v=h0qhZK0eGZY) video



|Base Network                       | Person | Car  | Bus | Bicycle | Motorbike | Train | Aeroplane | AP |
| --------------------------------- | :----: | :--: | :---: | :---: | :---: | :---: | :---: | :---: |
|CaffeNet                           | 49.36 | 57.81 | 57.39 | 59.25 | 60.43 | 62.19 | 52.62 | 44.61 |
|SqueezeNet                         | 62.33 | 69.94 | 64.60 | 68.02 | 66.85 | 69.63 | 63.02 |       |
|Resnet-18                          | 74.57 | 80.77 | 79.42 | 76.97 | 78.16 | 79.20 | 71.18 | 68.37 |
|VGG16_reduced                      | ----- | ----- | ----- |-------|---------|-----  |---------|------|

*you can download weights via this [link](https://goo.gl/Uwyom7) 

