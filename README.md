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

~~1.serializing model will cost a lot of time, if could save the result of serializing model to disk, it will be more convenient~~

2.I think the cost of inference is not good,I want to change the architecture.If I succeeded,I will push it

3.I found that same code get different result in gtx1080 and tx2. WHY?????



---

There is a problem，the code runs on different platforms, get different result.

For example, there are 4 pictures, detected result.

**The first ，GTX1080_mobileNetSSD_Caffe**

![image](https://github.com/Ghustwb/MobileNet-SSD-TensorRT/blob/master/testPic/GTX1080_mobileNetSSD_Caffe.jpg)

**The second，GTX1080_mobileNetSSD_TensorRT3.0.4**

![image](https://github.com/Ghustwb/MobileNet-SSD-TensorRT/blob/master/testPic/GTX1080_mobileNetSSD_TensorRT3.0.4.jpg)

**The third,TX2_mobileNetSSD_Caffe**

![image](https://github.com/Ghustwb/MobileNet-SSD-TensorRT/blob/master/testPic/TX2_mobileNetSSD_Caffe.jpg)

**The third,TX2_mobileNetSSD_TensorRT3.0.4**

![image](https://github.com/Ghustwb/MobileNet-SSD-TensorRT/blob/master/testPic/TX2_mobileNetSSD_TensorRT3.0.4_Jetpack3.2.jpg)

**The premise of getting the above results is that the same code, the same caffemodel, the same input image**

As can be seen from the four pictures,

1、The results on the caffe are the same.

2、Same hardware platform,the result of tensorRT is different with Caffe's.

3、The accuracy on gtx1080 is higher than on tx2. So,Why??

Why the result of  tensorRT on TX2 is so bad???

