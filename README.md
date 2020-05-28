## The COMP6248 Reproducibility Challenge
Pytorch implement of [Pay Attention to Features, Transfer Learn Faster CNNs](https://openreview.net/pdf?id=ryxyCeHtPB)
## Team members:
Youhong Li, Jiewei Chen, Xuelian Yao [yl41n19,jc17n19,xy1m19@soton.ac.uk]
## Introduction
A common example of transfer learning is to train a model on a large data set, and then using the regularization methods on the target data to set fine-tune the pre-training weights. In order to explore the two questions of which neurons are available for source knowledge transfer and the importance of the target model in practice, the article proposes to establish a new model: AFDS (Attention Feature Extraction and Selection), which is mainly aimed on training small neural network. AFD (Attention Feature Distillation) is a regularizer that learning the importance of each channel in the output activation. AFS (Selection of Attention Features) can understand the importance of each channel in the output of the ConvBN layer and it is consists of a global average pool, and then creating a fully connected (FC) layer after each ConvBN layer in the source model. Finally, the authors used AFD regularization to fine-tune the target model on the target dataset to obtain the target model. 
## High-level overview of AFDS
In this implement, we chose to use pre-trained ResNet101 on ImageNet as source model, which is different from the original in this chart.
![image](https://github.com/uhomelee/DeepLearningCourseWork/blob/master/pic/2.png)
## Dependencies
- Python 3.7
- PyTorch 1.3.0
This implementation only supports running with GPUs.(need around 8GB memeory with `batch_size=16` )
## Dataset
In original paper, authors evaluated the methods with 6 diferent benchmark datasets: Caltech-256 (Grin et al., 2007) of 256 general object categories; Stanford Dogs 120 (Khosla et al., 2011) specializes to images containing dogs; MIT Indoors 67 (Quattoni & Torralba, 2009) for indoor scene classiﬁcation; Caltech-UCSD Birds-200-2011 (CUB-200-2011) (Wah et al., 2011) for classifying birds; and Food-101 (Bossard et al., 2014) for food categories. Example of 'Standord Dog' dataset:
- `load.py `，this project implement a function, `def load_datasets(set_name, input \_ size):`, given the name of dataset to return a `Dataloader` class for `Dataset` class in `train.py`. 
- `stanford_dogs_data.py`. Import the dataset online; traverse the folder and do `transforms.Resize`(Reseting image resolution), `transforms.Normalize`(normalizing the data by channel) et al. on each image; return `Dataloader` of training set and testing set and classes number.


**Arguments:**




