# The COMP6248 Reproducibility Challenge
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
## Dataset
We evaluate the methods with 6 diferent benchmark datasets: Caltech-256 (Grin et al., 2007) of 256 general object categories; Stanford Dogs 120 (Khosla et al., 2011) specializes to images containing dogs; MIT Indoors 67 (Quattoni & Torralba, 2009) for indoor scene classiﬁcation; Caltech-UCSD Birds-200-2011 (CUB-200-2011) (Wah et al., 2011) for classifying birds; and Food-101 (Bossard et al., 2014) for food categories. Example of 'Standord Dog' dataset:
- Run `stanford_dogs_data.py`. Importing the dataset online, preprocessing the dataset, and dividing it into test set and training set according to the generated labels.
- Run `load.py `，This project implement a function, `def load_datasets(set_name, input \_ size):`, given the name of dataset to return a `Dataloader` class. 
- Run `load_caltech256.py`, this file specifically download Caltech256 dataset, and split training and testing set. Use the code 
``` train_data, test_data, num_class = load_data() ``` to get training dataset, testing dataset and the number of classes.
- `transforms.Resize`：Reseting image resolution.
-  `transforms.Normalize`, normalizing the data by channel, that is, first subtract the mean and then divide by the standard deviation.

## Training

 `python train.py`

**Arguments:**
-  `data_dir`: The directory of download data
-  `save_model`: The directory of saved model
-  `base_model`: Choose the base model, and the choices are `resnet101`, `resnet50`.
-  `data_dir`: The directory of download data
-  `lr_init `: The learning rate of initialization. default =  `0.01 `.
-  `max_iter `: Maximum number of iterations for SGD, int type, default = ` 4500 `.
-  `image_size `：Scaling images to this size before cropping.
-  `batch_size `: Batch size. default=  `48 `
-  `lr_scheduler `: The module provides some methods to adjust the learning rate based on the number of epoch trainings. Under normal     circumstances, with the increase of epoch, the learning rate is gradually reduced to achieve a better training effect.
-  `reg_type`: The type of regularization. The choices are `l2_sp`,`fea_map`,`afd`
-  `lambda_afd `: Used for AFD， default =  `0.01 `
-  `ratio `:  Used as `d` for winner take all function.default =  `0.9`.
-  `thres_s `: threshold for mask s
-  `thres_m `: threshold for mask m
## Team members:
Youhong Li [yl41n19@soton.ac.uk]

Jiewei Chen [jc17n19@soton.ac.uk]

Xuelian Yao [xy1m19@soton.ac.uk]
