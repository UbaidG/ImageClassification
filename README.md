# ImageClassification
Image Classification using ANN and CNN.

Using Tensorflow 2, keras, numpy, matplotlib and sklearn python lib.
```python
%tensorflow_version 2.x

import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , classification_report
```
### Loading Cifar10 dataset using load_data() from keras.datasets
```python
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
```
### Defining classes and fine tuning training parameters
```python
classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
y_train = y_train.reshape(-1,)
x_train = x_train/255
x_test = x_test/255
```
## Training model
```python
ann = keras.Sequential([
              keras.layers.Flatten(input_shape = (32,32,3)),
              keras.layers.Dense(3000, activation = 'relu'),
              keras.layers.Dense(1000, activation = 'relu'),
              keras.layers.Dense(10, activation = 'sigmoid')
])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(x_train, y_train, epochs = 2)
```
### Checking stats of the network
```python
y_pred = ann.predict(x_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))
```
