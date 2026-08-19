# CNN Architectures Using Transfer Learning

This experiment studies the evolution of CNN architectures and implements transfer learning using a pretrained VGG16 model on the CIFAR-10 dataset. The experiment includes model training, fine-tuning, performance evaluation, and comparison of CNN architectures.

### Dataset

CIFAR-10 contains:

* 50,000 training images
* 10,000 testing images
* 10 classes
* Image size: 32 × 32 × 3

Classes include Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

### Transfer Learning

A pretrained VGG16 model with ImageNet weights is used as the feature extractor. Its original classifier is removed, the convolutional layers are initially frozen, and a new classifier is added for CIFAR-10 classification.

### Model Configuration

| Parameter          | Value                     |
| ------------------ | ------------------------- |
| Model              | VGG16                     |
| Pretrained Weights | ImageNet                  |
| Optimizer          | Adam                      |
| Learning Rate      | 0.001                     |
| Batch Size         | 32                        |
| Epochs             | 10                        |
| Loss               | Categorical Cross Entropy |
| Metric             | Accuracy                  |
| Dense Units        | 256                       |
| Output Classes     | 10                        |

### Implementation

```python
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(32, 32, 3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(10, activation="softmax")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)
```

### Evaluation

The trained model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Classification Report

Fine-tuning is performed by unfreezing the last convolutional block and training for an additional 5–10 epochs.

### CNN Architectures Compared

| Architecture | Contribution                   |
| ------------ | ------------------------------ |
| LeNet-5      | First practical CNN            |
| AlexNet      | ReLU, Dropout and GPU training |
| VGG16        | Deep network using 3×3 filters |
| GoogleNet    | Inception modules              |
| ResNet       | Residual learning              |

### References

* LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998.
* Krizhevsky et al., *ImageNet Classification with Deep Convolutional Neural Networks*, 2012.
* Simonyan and Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, 2015.
* Szegedy et al., *Going Deeper with Convolutions*, 2015.
* He et al., *Deep Residual Learning for Image Recognition*, 2016.
* TensorFlow and Keras Documentation.
