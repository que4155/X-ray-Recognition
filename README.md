# X-ray-Recognition
Recognize sex by  pubic symphysis in KUB x-ray image using transfer learning with Inception Resnet V2

# Difference between Female and Male

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
 
# Predictions
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
Number of female image : 169

Number of male image : 180

# Steps
Download pretrained model Inception ResNet V2 and unzip it to ckpt file

Classify your Data and save a Label.txt, i even reshape them to make sure the pubis symphysis right in center of picture

Using tfrecord.py to generate tfrecord for training

Using train.py to transfer learning with Inception Resnet V2 and some fine tuning, then we can get new model

Using evaluate.py and last model to evaluate, here i tested this model with 50 pictures



# References
https://kwotsin.github.io/post/transfer_learning/

https://my.oschina.net/caibobit/blog/1605196
