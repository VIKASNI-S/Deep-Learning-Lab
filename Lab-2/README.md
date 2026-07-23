## Multi-Layer Perceptron (MLP) for Fashion-MNIST Classification with Hyperparameter Optimization

## Overview

This experiment demonstrates the implementation of a Multi-Layer Perceptron (MLP) for multi-class image classification using the Fashion-MNIST dataset. The model is trained using TensorFlow/Keras and evaluated using standard classification metrics. Hyperparameter optimization is performed using RandomizedSearchCV with SciKeras to improve the model's performance.

---

# Dataset Information

### Dataset Name
Fashion-MNIST

### Source
Zalando Research

https://github.com/zalandoresearch/fashion-mnist

### Description

Fashion-MNIST is a benchmark dataset consisting of grayscale images of fashion products. It is designed as a drop-in replacement for the original MNIST handwritten digit dataset.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Training Images | 60,000 |
| Testing Images | 10,000 |
| Image Size | 28 × 28 pixels |
| Image Type | Grayscale |
| Number of Classes | 10 |


# Methodology

The experiment consists of the following stages:

1. Dataset Exploration
2. Data Preprocessing
3. Model Construction
4. Model Training
5. Model Evaluation
6. Hyperparameter Optimization
7. Performance Comparison

---

# Model Architecture

Input Layer

- 784 Neurons

Hidden Layer 1

- Dense Layer
- 128 Neurons
- ReLU Activation

Hidden Layer 2

- Dense Layer
- 64 Neurons
- ReLU Activation

Output Layer

- Dense Layer
- 10 Neurons
- Softmax Activation

---

# Hyperparameter Optimization

RandomizedSearchCV was used to optimize:

- Hidden Layers
- Hidden Neurons
- Activation Function
- Optimizer
- Learning Rate
- Batch Size
- Epochs
- Dropout Rate

---

# Dependencies

Install the following packages before running the project.

```
Python >= 3.10
TensorFlow
NumPy
Matplotlib
Pandas
Scikit-learn
SciKeras
```

Install all dependencies using:

```bash
pip install tensorflow
pip install numpy
pip install matplotlib
pip install pandas
pip install scikit-learn
pip install scikeras
```

or

```bash
pip install -r requirements.txt
```

---

# Outputs Generated

The program generates:

- Sample Images
- Class Distribution Plot
- Training Accuracy Plot
- Validation Accuracy Plot
- Training Loss Plot
- Validation Loss Plot
- Confusion Matrix
- Hyperparameter Search Plot
- Best Model Accuracy Comparison

---

# Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

# Results

The optimized model is compared with the baseline model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Training Time

The best hyperparameters obtained after optimization are also reported.

---

# Conclusion

The experiment demonstrates the implementation of a Multi-Layer Perceptron for image classification using the Fashion-MNIST dataset. Hyperparameter optimization helps identify an effective combination of model parameters and provides insight into their impact on classification performance.

---
