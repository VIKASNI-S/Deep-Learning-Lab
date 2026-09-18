CNN Training and Optimization on Oxford-IIIT Pet Dataset
Overview

This project studies CNN training and optimization using the Oxford-IIIT Pet Dataset. Images are resized to 224×224×3 for multi-class pet breed classification.

Experiments

The project evaluates:

Weight initialization

Regularization

Batch Normalization

Dropout

Optimization algorithms

CNN hyperparameters

Transfer learning

MobileNetV2 fine-tuning

5-fold cross-validation

Methodology

Different CNN configurations are trained and evaluated using validation performance. MobileNetV2 is used for transfer learning and fine-tuning. The best configuration is selected using 5-fold cross-validation and then evaluated on an independent test set.

Dataset

Oxford-IIIT Pet Dataset — 37 cat and dog breed classes.

Technologies

Python

TensorFlow/Keras

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

Evaluation

Models are evaluated using:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

Results

The final results and comparisons are provided in the results/ directory.
