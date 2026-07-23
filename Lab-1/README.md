# Single Layer Perceptron for Banknote Authentication

## Overview

This project implements a **Single Layer Perceptron (SLP)** from scratch using Python and NumPy for binary classification. The objective is to classify banknotes as **genuine** or **forged** using statistical features extracted from images of banknotes.

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

