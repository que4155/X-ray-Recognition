# X-ray-Recognition
Recognize sex by  pubic symphysis in x-ray image using transfer learning with Inception Resnet V2

# Where is Different between Two Sex - the angle

![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/19y_m_0581123.jpg)

     Male has a small angle
![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/20y_f_1865012.jpg)
     
    Female generally has bigger angle
# Reseult in test
 Fifty image for test
 
 batch size : 10
 
 epoch : 20
 
 Accuracy : 0.959
 ![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/ac.png)
 
# Some Predictions
 ![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/predict.png)

# Environment 
Pretrained Model : inception Resnet V2 2016_08_30

Python : 3.6.4

Tensorflow-GPU : 1.6.0

GPU : NVidia GTX 860M

NVidia CUDA : 9.1

Epochs : 100

Batch Size : 4

Learning Rate : 0.0001

Learning Rate Decay Factor : 0.7
# DataSet
num of female image : 169

num of male image : 180

I were a Radiologist intern in Show Chwan Memorial Hospital,Taiwan from 2018/7~2019/1,used my free time and got permission to collect data which is only JPEG file without any personal tags.

# Steps
Download pretrained model Inception V2 and unzip it to ckpt file

Classify your Data,i even reshape them to make sure the pubis symphysis right in center of picture

Use tfrecord.py to generate tfrecord for training

Use train.py to train with Inception Resnet V2 ,a combinition structure of VGG an Resnet, then we can get new model

Use evaluate.py and last model to get evaluation , here i test this model with 50 pictures



# References
https://kwotsin.github.io/post/transfer_learning/

https://my.oschina.net/caibobit/blog/1605196
