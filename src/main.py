from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

print(keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

for i in range(4):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(X_train[i + 100], cmap=pyplot.get_cmap("gray"))
    pyplot.show()

X_train = X_train_full[5000:] / 255.0
X_valid = X_train_full[:5000] / 255.0
y_train = y_train_full[5000:]
y_valid = y_train_full[:5000]

X_test = X_test / 255.0


model = keras.model.Sequential(
    [
        layers.Flatten(input_shape=[28, 28]),
        layers.Dense(300, activation="relu"),
        layers.Dense(100, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
