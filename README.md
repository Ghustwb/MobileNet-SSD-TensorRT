# MobileNet-SSD-TensorRT
To accelerate mobileNet-ssd with tensorRT

TensorRT-Mobilenet-SSD can run 50fps on jetson tx2

**Requierments:**

1.tensorRT3.0.4

2.cudnn7

3.opencv

**Reference:**

https://github.com/saikumarGadde/tensorrt-ssd-easy

https://github.com/chuanqi305/MobileNet-SSD

I replaced depthwise with group_conv,because group_conv  has been optimized in cudnn7

I retrianed mobileNet-SSD,my number of classfication is 4

If you coding with QtCreator,you could run it directly

**TODO:**

1.serializing model will cost a lot of time, if could save the result of serializing model to disk, it will be more convenient

2.I think the cost of inference is not good,I want to change the architecture.If I succeeded,I will push it