# X-ray-Recognition
Recognize sex by  pubic symphysis in KUB x-ray image by deep learning with Inception Resnet V2

# Difference between Female and Male

![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/19y_m_0581123.jpg)

     Male has a small angle
![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/20y_f_1865012.jpg)
     
    Female generally has bigger angle
# Reseult and Predictions
 Fifty image for test
 
 Accuracy : 0.959
 ![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/ac.png)
 
 Predictions
 ![image](https://github.com/que4155/X-ray-Recognition/blob/master/picture/predict.png)

# Environment 
Pretrained Model : Inception Resnet V2 2016_08_30

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
Download pretrained model Inception ResNet V2 and unzip it and thus you have a "checkpoint_file" path.

Classify your data in different folders and then save a "Label.txt" which include your label name and its code name, and thus we have a "dataset_dir" path.

Using tfrecord.py to generate tfrecord from dataset for training.

Using train.py to learning with Inception Resnet V2, but you need to setup "log_dir" first to save records.

Using evaluate.py and last model to test and then open tensorboard with log folder for inspection.



# References
https://kwotsin.github.io/post/transfer_learning/

https://my.oschina.net/caibobit/blog/1605196
