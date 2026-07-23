# Single Layer Perceptron for Banknote Authentication

## Overview

This project implements a **Single Layer Perceptron (SLP)** from scratch using Python and NumPy for binary classification. The objective is to classify banknotes as **genuine** or **forged** using statistical features extracted from images of banknotes.

The implementation follows the Deep Learning Laboratory experiment requirements and includes:

- Dataset Exploration
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Perceptron Implementation from Scratch
- Model Training
- Model Evaluation
- Visualization of Learning Process

---

# Project Structure

```
Single_Layer_Perceptron/
│
├── Dataset/
│   ├── banknote+authentication.zip
│   └── Data_banknote_authentication.txt
│
├── Figures/
│   ├── Histograms.eps
│   ├── Correlation_Heatmap.eps
│   ├── Scatter_Plot.eps
│   ├── Learning_Curve.eps
│   ├── Weight_Evolution.eps
│   ├── Bias_Evolution.eps
│   ├── Confusion_Matrix.eps
│   └── Learning_Rate_Comparison.eps
│
├── perceptron.py
├── requirements.txt
├── README.md
└── Report.pdf
```

---

# Objective

The objective of this experiment is to design and implement a **Single Layer Perceptron** from scratch for binary classification. The perceptron is trained using the Perceptron Learning Rule and evaluated on the Banknote Authentication dataset.

---

# Dataset Information

## Dataset Name

**Banknote Authentication Dataset**

## Source

UCI Machine Learning Repository

https://archive.ics.uci.edu/ml/datasets/banknote+authentication

---

## Dataset Description

The dataset consists of statistical features extracted from images of genuine and forged banknotes using Wavelet Transform techniques.

Each sample contains four numerical features and one target class.

---

## Dataset Statistics

| Property | Value |
|-----------|-------|
| Total Samples | 1372 |
| Features | 4 |
| Target Classes | 2 |
| Missing Values | None |
| Classification Type | Binary Classification |

---

## Features

| Feature | Description |
|----------|-------------|
| Variance | Variance of Wavelet Transformed Image |
| Skewness | Skewness of Wavelet Transformed Image |
| Curtosis | Curtosis of Wavelet Transformed Image |
| Entropy | Entropy of Image |
| Class | 0 = Genuine, 1 = Forged |

---

# Experimental Workflow

The project consists of the following stages:

### 1. Dataset Exploration

- Load dataset
- Display first five samples
- Check dataset dimensions
- Identify missing values
- Display statistical summary

---

### 2. Exploratory Data Analysis

Generated visualizations include:

- Feature Histograms
- Correlation Heatmap
- Scatter Plot
- Boxplots

---

### 3. Data Preprocessing

- Feature normalization using StandardScaler
- Dataset split into
  - 80% Training Data
  - 20% Testing Data

---

### 4. Perceptron Implementation

Implemented from scratch without using sklearn's Perceptron.

Includes:

- Weight Initialization
- Bias Initialization
- Step Activation Function
- Forward Propagation
- Perceptron Learning Rule

---

### 5. Model Training

During each epoch the following are displayed:

- Epoch Number
- Misclassified Samples
- Updated Weights
- Updated Bias

---

### 6. Model Evaluation

Performance metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

### 7. Mandatory Plots

The project generates the following plots:

- Feature Histograms
- Correlation Heatmap
- Scatter Plot
- Training Error vs Epoch
- Weight Evolution
- Bias Evolution
- Confusion Matrix
- Learning Rate Comparison

All plots are saved in **EPS format** for inclusion in LaTeX reports.

---

# Dependencies

Install the required Python libraries before running the project.

```
numpy
pandas
matplotlib
seaborn
scikit-learn
```

---

# Installing Dependencies

Using pip

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

or

```bash
pip install -r requirements.txt
```

---

# Execution Instructions

## Step 1

Clone the repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
```

---

## Step 2

Navigate to the project folder

```bash
cd Single_Layer_Perceptron
```

---

## Step 3

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## Step 4

Place the dataset

```
banknote+authentication.zip
```

inside the **Dataset** folder.

---

## Step 5

Extract the ZIP file or use the code provided in the notebook to automatically extract it.

The extracted file should be

```
Data_banknote_authentication.txt
```

---

## Step 6

Run the Python program

```bash
python perceptron.py
```

or execute the notebook cells sequentially if using Jupyter Notebook or Google Colab.

---

# Expected Outputs

The program produces:

- Dataset Summary
- Descriptive Statistics
- Histograms
- Heatmap
- Scatter Plot
- Boxplots
- Training Progress
- Weight Updates
- Bias Updates
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Learning Rate Comparison
- EPS figures for report generation

---

# Results

The implemented Single Layer Perceptron successfully classifies genuine and forged banknotes. During training, the number of misclassified samples generally decreases across epochs, demonstrating effective learning. Minor fluctuations in training error are expected due to iterative weight updates. The evaluation metrics indicate good classification performance on unseen test data.

---

# Future Enhancements

Possible improvements include:

- Multi-Layer Perceptron (MLP)
- Adaptive Learning Rate
- Cross Validation
- Hyperparameter Optimization
- Feature Selection
- Comparison with other Machine Learning Algorithms

---

# Authors

**VIKASNI S**

B.Tech Artificial Intelligence and Data Science

Shiv Nadar University Chennai

---

# License

This project is developed for academic and educational purposes.
