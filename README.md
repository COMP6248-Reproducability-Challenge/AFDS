# University of Southampton
## The COMP6248 Reproducibility Challenge
Pytorch implement of [Pay Attention to Features, Transfer Learn Faster CNNs](https://openreview.net/pdf?id=ryxyCeHtPB)
## Introduction
A common example of transfer learning is to train a model on a large data set, and then using the regularization methods on the target data to set fine-tune the pre-training weights. In order to explore the two questions of which neurons are available for source knowledge transfer and the importance of the target model in practice, the article proposes to establish a new model: AFDS (Attention Feature Extraction and Selection), which is mainly aimed on training small neural network. AFD (Attention Feature Distillation) is a regularizer that learning the importance of each channel in the output activation. AFS (Selection of Attention Features) can understand the importance of each channel in the output of the ConvBN layer. Our main work is to deploy and update the AFS model in ResNet-101. AFS consists of a global average pool, and then creating a fully connected (FC) layer after each ConvBN layer in the source model.  Finally, we use AFD regularization to fine-tune the target model on the target dataset to obtain the target model. AFDS is deployed on ResNet-101 and the latest calculation is simplified, and we use three data sets to train the model, there are Stanford Dogs 120, MIT Indoor 67 and Caltech-256 and other extensive data. Under these datasets, the AFDS model still maintains a high task accuracy.
## High-level overview of AFDS
We used [this](https://github.com/uhomelee/DeepLearningCourseWork) project as baseline.

![image](https://github.com/uhomelee/DeepLearningCourseWork/blob/master/pic/1.png)
![image](https://github.com/uhomelee/DeepLearningCourseWork/blob/master/pic/2.png)
## Dependencies
- Python 2.7
- PyTorch 
This implementation only supports running with GPUs.
##


## Team members:
Youhong Li [yl41n19@soton.ac.uk]

Jiewei Chen [jc17n19@soton.ac.uk]

Xuelian Yao [xy1m19@soton.ac.uk]
