# Fashion Item Recognition – Deep Learning Week 3

## About This Repository

This repository contains my Week 3 Deep Learning assignment and mini-project completed as part of the **Skill Nexis Deep Learning Program**.

The main objective is to understand the fundamentals of deep learning and apply Convolutional Neural Networks (CNNs) to image classification using the Fashion-MNIST dataset.

## Project Overview

**Project Title:** Fashion Item Recognition

Fashion Item Recognition is an image classification project that uses deep learning to identify different categories of clothing from images.

The model takes a clothing image as input and predicts its category from 10 different fashion classes.

## Dataset

**Fashion-MNIST Dataset**

- 70,000 grayscale images.
- Image dimensions: 28 × 28 pixels.
- 60,000 training images.
- 10,000 testing images.
- 10 different clothing categories.

### Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Assignment – Topics Covered

The Week 3 assignment covers the following concepts:

- Images represented as tensors.
- Convolution and filters.
- Padding and pooling operations.
- CNN architecture and its working.
- Image normalization.
- Dataset loading and preprocessing.
- Comparison between CNN and fully connected neural networks.
- Classification error analysis.

## Mini-Project Implementation

The mini-project includes the following steps:

1. Loading the Fashion-MNIST dataset.
2. Visualizing sample fashion images.
3. Normalizing image pixel values.
4. Splitting the dataset into training and validation sets.
5. Building a Fully Connected Neural Network as a baseline model.
6. Building a Convolutional Neural Network (CNN).
7. Training and evaluating both models.
8. Comparing model performance.
9. Generating predictions on test images.
10. Analyzing misclassified examples.

## Models Developed

### 1. Fully Connected Neural Network

A basic neural network is developed as a baseline model using:
- Flatten layer
- Dense layer with ReLU activation
- Output layer with Softmax activation

### 2. Convolutional Neural Network (CNN)

A CNN is developed to extract image features and classify fashion items using:
- Convolutional layers
- ReLU activation
- Max Pooling layers
- Flatten layer
- Dense layer
- Softmax output layer

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pandas
- Google Colab
- GitHub

## Repository Structure

```text
fashion-item-recognition-week3/
│
├── assignment/
│   └── assignment_notebook.ipynb
│
├── mini-project/
│   ├── Fashion_Item_Recognition.ipynb
│   ├── fashion_mnist_cnn.keras
│   ├── fashion_mnist_baseline.keras
│   ├── fashion_model_comparison.csv
│   ├── fashion_misclassification_analysis.csv
│   ├── prediction_gallery.png
│   └── five_misclassified_examples.png
│
└── README.md
```

*Note: The file names and folder structure can be adjusted to match the files uploaded to this repository.*

## Results and Evaluation

Both the Fully Connected Neural Network and CNN are evaluated using the Fashion-MNIST test dataset.

The evaluation includes:
- Test accuracy
- Test loss
- Training and validation accuracy
- Training and validation loss
- Prediction results
- Misclassification analysis

A comparison table is generated to understand the performance differences between the two models.

## Learning Outcomes

Through this assignment and mini-project, I gained practical understanding of:

- Image preprocessing and normalization.
- How convolutional layers extract image features.
- The role of pooling in CNNs.
- Building and training deep learning models.
- Comparing different neural network architectures.
- Evaluating classification performance.
- Understanding and analyzing incorrect predictions.

## Conclusion

This assignment and mini-project provided practical experience in applying deep learning techniques to image classification. By developing and comparing a Fully Connected Neural Network and a CNN, I gained a better understanding of how CNN architectures are used for image recognition tasks.

---

**Program:** Skill Nexis – Deep Learning  
**Week:** Week 3  
**Project:** Fashion Item Recognition  
**Domain:** Deep Learning / Computer Vision