# Lab 1:
## Objective:
The objective of this project is to build a Convolutional Neural Network (CNN) model using PyTorch to classify images of handwritten digits from the MNIST dataset. The model is trained to recognize digits (0-9) and evaluate its performance on test data.
## Libraries and Tools Used:
PyTorch: For building the neural network and handling the training process.
Torchvision: For image transformations (like normalization and tensor conversion).
NumPy: For handling data and manipulating arrays.
PIL (Python Imaging Library): For loading and processing image data.
Struct: For reading binary file formats specific to MNIST dataset files.
## Dataset Transformation:
The images are transformed using the following steps:

Convert to Tensor: transforms.ToTensor() converts the images from PIL format to PyTorch tensors.
Normalization: transforms.Normalize((0.5,), (0.5,)) normalizes the pixel values to be between -1 and 1 for improved model convergence.

## CNN Model Architecture:
### Convolutional Layers:
conv1: 32 filters, 3x3 kernel.
conv2: 64 filters, 3x3 kernel.
conv3: 128 filters, 3x3 kernel.
### Fully Connected Layers:
fc1: Fully connected layer with 512 neurons.
fc2: Output layer with 10 neurons (one for each digit class).

## Training:
Loss Function: CrossEntropyLoss, suitable for multi-class classification tasks.
Optimizer: Adam optimizer with a learning rate of 0.001.
## Conclusion:
The model is trained on the MNIST dataset, which consists of images of handwritten digits. After training, the model's accuracy on the test set is evaluated. This model can be further enhanced by experimenting with different architectures, data augmentation techniques, or hyperparameters. The model can be applied to digit recognition tasks, such as OCR (Optical Character Recognition), and is a fundamental starting point for understanding CNNs and deep learning in image processing.
