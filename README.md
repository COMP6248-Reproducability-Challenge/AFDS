# University of Southampton
## The COMP6248 Reproducibility Challenge
Pytorch implement of [Pay Attention to Features, Transfer Learn Faster CNNs](https://openreview.net/pdf?id=ryxyCeHtPB)
## Introduction
A common example of transfer learning is to train a model on a large data set, and then using the regularization methods on the target data to set fine-tune the pre-training weights. In order to explore the question of which neurons are available for source knowledge transfer and the importance of the target model in practice, the article proposes to establish a new model: Attention Feature Extraction and Selection (AFDS), which is mainly aimed at training Small neural network. Attention feature distillation AFD is a regularizer that learns the importance of each channel in the output activation. AFS is the selection of attention features and understands the importance of each channel in the output of the ConvBN layer. We The main work is to deploy and update the AFS model in ResNet-101. AFS consists of a global average pool, and then create a fully connected (FC) layer after each ConvBN layer in the source model. Finally, use AFD regularization on the target data set Fine-tune the target model to get the target model. AFDS is deployed on ResNet-101 and the latest calculation is simplified, and we use three data sets to train the model, in Stanford Dogs 120, MIT Indoor 67 and Caltech-256 and other extensive data Under the set, the AFDS model still maintains a high task accuracy.



![image](https://github.com/uhomelee/DeepLearningCourseWork/blob/master/pic/1.png)
![image](https://github.com/uhomelee/DeepLearningCourseWork/blob/master/pic/2.png)


## Team members:
Youhong Li [yl41n19@soton.ac.uk]

Jiewei Chen [jc17n19@soton.ac.uk]

Xuelian Yao [xy1m19@soton.ac.uk]
