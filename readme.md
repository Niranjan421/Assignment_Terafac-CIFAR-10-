# CIFAR-10 Image Classification using Deep Learning

This project implements a deep learning-based image classification system using the CIFAR-10 dataset. The goal is to accurately classify images into one of ten object categories using Convolutional Neural Networks (CNNs) and transfer learning techniques.

---

## Problem Understanding

Image classification is a fundamental task in computer vision where a model learns to assign a label to an input image. The CIFAR-10 dataset presents a challenging classification problem due to:

- Low image resolution (32×32 pixels)
- High intra-class variation
- Similar visual patterns across different classes

The dataset contains 10 classes:
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

The objective is to design a model that can:
- Learn meaningful visual features from images
- Generalize well to unseen data
- Achieve high classification accuracy

To address this, the project uses transfer learning with a pre-trained ResNet50 model and applies data augmentation and regularization techniques to improve performance.

---

## Dataset

- Dataset Name: CIFAR-10
- Total Images: 60,000
- Training Images: 50,000
- Testing Images: 10,000
- Image Size: 32×32 RGB
- Number of Classes: 10

The dataset is automatically loaded using deep learning frameworks.

---

## Objectives

- Load and preprocess the CIFAR-10 dataset
- Build a CNN-based classification model
- Apply transfer learning for better feature extraction
- Improve generalization using data augmentation
- Evaluate model performance using standard metrics
- Visualize predictions and misclassified samples

---

## Tools and Technologies

- Programming Language: Python
- Deep Learning Framework: TensorFlow / Keras
- Libraries: NumPy, Pandas, Matplotlib
- Dataset: CIFAR-10
- IDE / Platform: Google Colab / Jupyter Notebook / VS Code

---

## Methodology

### Level 1: Baseline Model
- Used pre-trained ResNet50 as a feature extractor
- Froze base layers and trained custom classification layers
- Evaluated performance using accuracy and loss curves

### Level 2: Intermediate Improvements
- Applied data augmentation (rotation, flipping, zooming)
- Used dropout and L2 regularization
- Tuned hyperparameters such as learning rate and batch size

### Level 3: Model Evaluation and Analysis
- Generated confusion matrix and class-wise accuracy
- Visualized training and validation curves
- Analyzed misclassified images

---

## Results

- Improved classification accuracy using transfer learning
- Reduced overfitting through data augmentation
- Identified class-wise performance using confusion matrix
- Visual insights into model errors

---

## Setup Instructions

Prerequisites: Python 3.8+, pip, Git, Jupyter Notebook / VS Code / Google Colab

Check Python version:
python --version

Clone the repository:
git clone https://github.com/Niranjan421/Assignment_Terafac-CIFAR-10-

Create virtual environment:
python -m venv venv

Activate virtual environment (Windows):
venv\Scripts\activate

Activate virtual environment (Linux/macOS):
source venv/bin/activate

Install dependencies:
pip install -r requirements.txt

If requirements.txt is not available:
pip install tensorflow numpy pandas matplotlib scikit-learn

Dataset setup:
CIFAR-10 dataset is downloaded automatically when the code runs. No manual download required.

Run the project using Jupyter Notebook:
jupyter notebook

Or run using Python script:
python main.py

Optional GPU support (local system):
pip install tensorflow-gpu

Google Colab GPU:
Runtime → Change runtime type → GPU

Deactivate virtual environment (optional):
deactivate

cd cifar10-image-classification

