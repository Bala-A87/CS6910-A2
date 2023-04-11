# CS6910-A2
Programming assignment 2 of CS6910 - Fundamentals of Deep Learning

## [Part A](./Part-A/)

Goal: Implement a convolutional neural network using PyTorch and train it from scratch. The architecture of the network allows 5 convolution-activation-maxpool (and optionally batchnorm) layers, with flexibility on the number of convolutional filters and their filter sizes for each layer and the size of the last maxpool layer. The convolutional layers are followed by a two-layer (both fully-connected) classifier, with the output layer returning a probability distribution over 10 classes, to support the *iNaturalist* dataset. The width of the other dense layer can be chosen, with an optional dropout following it. Activation functions used in the convolutional layers and the dense layer can be given as arguments.

## [Part B](./Part-B/)

Goal: Load a pre-trained model and fine-tune it for the *iNaturalist* dataset. The architecture chosen is *ResNet50*. Fine-tuning is done by freezing the layers of the network and replacing the fully-connected output layer (of width 1000) by one of width 10, and training only that.

## Link to dataset

The dataset can be found [here](https://storage.googleapis.com/wandb_datasets/nature_12K.zip). In order to run the code, please download the zip folder, extract it and transfer its contents to a directory named `data` created here. [This link](./data/) should work once this is done.