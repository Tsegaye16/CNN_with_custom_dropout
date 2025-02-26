---

# Image Classification with Custom Dropout CNN

This project implements an image classification model using a Convolutional Neural Network (CNN) with a custom dropout regularization technique. The model is trained on a dataset of images categorized into different classes.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Model Architecture](#model-architecture)
6. [Custom Dropout](#custom-dropout)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## Introduction

This project demonstrates how to build and train a CNN for image classification using TensorFlow and Keras. It includes a custom implementation of the dropout technique to prevent overfitting. The model is trained on a dataset of images and evaluated on a separate test set.

## Features

- Custom dropout implementation for regularization.
- CNN architecture for image classification.
- Training and evaluation on a dataset of images.
- Prediction on unseen images.

## Requirements

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- Pillow
- scikit-learn

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tsegaye16/CNN_with_custom_dropout.git
   cd CNN_with_custom_dropout
   ```

2. **Install the required packages**:
   ```bash
   pip install tensorflow numpy pillow scikit-learn
   ```

3. **Prepare the dataset**:
   - Place your training images in the `data/seg_train/seg_train` directory, organized into subdirectories for each category.
   - Place your test images in the `data/seg_test/seg_test` directory, organized similarly.
   - Place your unseen images for prediction in the `data/seg_pred/seg_pred` directory.



2. **Observe the output**:
   - The script will train the CNN model on the training dataset and evaluate it on the test dataset.
   - It will then make predictions on the unseen images and print the predicted categories.

## Model Architecture

The CNN model consists of the following layers:

- Convolutional layer with 32 filters, ReLU activation, and max pooling.
- Convolutional layer with 64 filters, ReLU activation, and max pooling.
- Flatten layer to convert the 2D matrices to a vector.
- Fully connected layer with 128 units and ReLU activation.
- Output layer with softmax activation for multi-class classification.

## Custom Dropout

The custom dropout function randomly sets a fraction of the input units to zero during training. This helps to prevent overfitting by ensuring that the model does not rely too heavily on any single unit. The dropout rate can be adjusted to control the fraction of units that are dropped.

## Results

The model's performance is evaluated using accuracy and loss metrics on the test dataset. The script prints the test loss and accuracy after training. Additionally, it prints the predicted categories for the unseen images.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your proposed changes.



---
