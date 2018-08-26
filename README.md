# X-ray-Recognition
Recognize sex by  pubic symphysis in x-ray image using transfer learning with Incepiton V2

# Where is Different between Two Sex

![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/19y_m_0581123.jpg)

     Male  
![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/20y_f_1865012.jpg)
     
    Female
# Reseult in test
 Fifty image for test
 
 batch size : 10
 
 epoch : 20
 
 Accuracy : 0.959
 ![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/ac.png)
 
# DataSet
num of female image : 169

num of male image : 180

I am a Radiologist intern in Show Chwan Memorial Hospital,Taiwan from 2018/7~2019/1,used my free time and get permission to collect data.

# Steps
Download pre-train model Inception V2 and unzip it to ckpt file

Use python tfrecord.py to generate tfrecord for training

Use python train.py to train with Inception V2 ,and then we can make new model

Use python evaluate.py and last model to get evaluation 
# References
https://kwotsin.github.io/tech/2017/02/11/transfer-learning.html

https://my.oschina.net/caibobit/blog/1605196
