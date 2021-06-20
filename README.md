# Histopathologic-Cancer-Detection

A CNN model made to detect the presence of cancer cell in the image. 

The dataset was downloaded from kaggle. 

Kaggle kernel was used to build the model.



Histopathology is the microscopic study of the tissues in order to study the evidence of disease. Histopathologic cancer detection refers to the examination of the biopsy after the specimens have been place onto the glass slide. Recent advancements in deep learning-based algorithms have shown significant improvements in histological analysis of lymph node metastasis. Still the accuracy is not increased because of the difficulty in the handling of the images. In this paper, we have proposed and evaluated a Convolution Neural Network (CNN) based deep learning model to detect Cancer metastasis on histopathology images of lymph nodes to potentially improve the accuracy and efficiency of the diagnosis.


Histopathologic cancer detection means to detect the presence of the metastatic cancer in the specimen of the tissue obtained from biopsy. The problem which is addressed 
involves classification of the slide images of the tumors. The image can be classified into two class labels either Malign or Benign. If there is a metastatic cancer cell in the image, it needsto be classified as malign else it is classified it as benign. Automated lymph node metastasis detection has a very significant potential to improve the efficiency and accuracy of pathologists for the diagnostic process in this field.In the last few years there have been significant improvements in the field of computer vision task using Convolutional Neural Networks [1]. Following this pattern, in recent years CNN based computer aided metastasis detection have been proposed [2,3,4].In this paper, we introduce a robust method to predict presence of lymph node metastasis from whole slide pathology images using Convolutional Neural Networks. The architecture used to train the model for the detection of the lymph node metastasis does not use any pre trained model like VGG16, NasNet, Inception or any other model. The architecture used to classify the images has been implemented from scratch using Keras


The research work in the field of histopathologic cancer detection using Convolutional neural network is very vast domain. There have been a number of methods which have been proposed over the years for detection of cancer in images using CNN. The methodology proposed in paper makes use of a novel architecture rather than using 
pre trained models. This helps to reduce the cost of computation and results in faster convergence model. The dataset used for the purpose also packs the clinically - relevant task of metastasis detection into binary classification which can be trained very easily on a single Graphical Processing Unit (GPU) along with good accuracy scores. 


The methodology used for detection of the lymph node metastasis from whole slide pathology images is done using Convolutional Neural Networks. The Convolution Neural Network just like any other neural network consists of neurons with weights and biases. The inputsreceived by the neuron aresubjected to weighted average, passed through an activation function and output is given. Unlike neural networks, which take vectors for input, the CNN takes a multi-channeled image. The CNN is able to capture both spatial and temporal 
features of an image using relevant filters. In other words, we can use CNN to understand the sophistication of images. CNN does the role of reducing the image into such formats which makes it easier to process, while still maintaining its features which are critical for a good accuracy. The layers involved in CNN are Convolution Layer, Pooling Layer and the Fully Connected Layer.Convolution Layer: Involved in the operation of convolution. It uses kernel/filter to extract the features from the image. 
Assuming an image of dimension 32x32x3 when applied with a kernel/filter of 5x5x1.A single pixel value of the resultant image is obtained by a dot product of the filter and a small 5x5x1 chunk of the image i.e. 5*5*1=75-dimensional dot product + bias.
Pixel Value = wTx + b
The result obtained by this dot product is a scalar value. The filter/kernel is then slided over the entire image and the resultant image is obtained.Pooling: Pooling layer is used to reduce the spatial size. This helps to reduce the number of parameters and the computation in the neural network. The pooling layer works 
independently on each feature map. Max pool takes the filter and stride size and returns the maximum pixel value in the maximum in that particular stride.dFully Connected (FC) Layer: FC Layers are used to detect specific global configurations of features detected by the lower layers in the neural net. Each neuron of the FC layers has its own weights. It connects every node from one layer to every node of another layer.Activation Layer: It is generally put at the end of a layer or in between the neural network. The main task of the activation layer is to decide whether the neuron gives output or not. It is a nonlinear transformation that is done on the input signal to send as an output to the next layer of neurons as input. RELU is the most used activation function which converts all the negative inputs to zero and the neuron does not get activated. This makes the computation more efficient, which results in fast convergence of the neural network.


The Dataset used for training the model is the PatchCamelyon (PCam) dataset [5,6] which is taken from Kaggle. The PatchCamelyon is a new and challenging image classification dataset. The dataset available on Kaggle does not contain duplicate images which are present in the original data due to its probabilistic sampling. The dataset used for 
training the neural network consisted of 220,025 colour images. Each image is of size 96x96x3px. The images of the dataset are annoted with a binary class label which indicates 
the presence of metastatic tissue in the image. Out of the total training data, the number of positive samplesis 89,117 and the number of negative samples is 130,908
The dataset is divided into a training set of 176,020 examples and a validation set of 44,005 examples. We have used another 57,458 images astestsample A positive label for the image indicates that the center 32x32px region contains at least one pixel of tumour tissue. The outer region is provided to enable the design of the models that do not use any zero padding.




I propose a novel convolution neural network architecture for detection of metastatic cancer tissues on histopathologic images of lymph nodes. We exploited various machine learning techniques in order to increase the accuracy as well as the computation efficiency of the neural network. The experimental results gave a very good accuracy for the testing example. One direction of future work could be implementation of Rotation invariant CNN for detection of the presence of lymph node metastasis.

