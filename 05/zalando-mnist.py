# Librerías Deep Learning
import sys
# import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten


print("\nVersiones:")
# print("- Python", sys.version)
print("- TensorFlow", tf.__version__)
print("- tf.Keras", tf.keras.__version__)

# Dataset MNIST, que está en storage.googleapis.com en TensorFlow/Keras
# mnist = tf.keras.datasets.mnist
fashion_mnist = tf.keras.datasets.fashion_mnist

# Defino dos tuplas, cada una con dos arrays numpy
# train: Tupla conjunto de entrenamiento
# test: Tupla conjunto de prueba
# En cada una, el primer elemento es una imagen y el segundo su etiqueta
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Lista con los nombres de las clases de ropa
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Preprocesado del dataset -> adaptar los datos para usarlos en la red neuronal
# Normalización: de uint8 [0 - 255] a float32 [0 - 1]
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Cambio de forma el tensor
# Convierto la matriz de 60000 elementos 28x28 a 60000 elemntos de 784 items
# Pasamos por tanto de 3D a 2D
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)

# Cambio las etiquetas a codificación one-hot
# Los valores pasan a un vector de 10 items, cada posición es un valor
# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train, num_classes=10)
# y_test = to_categorical(y_test, num_classes=10)

# Definición del modelo
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))
print("\n")
model.summary()
# model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

# Entrenamiento del modelo
print("\n")
model.fit(train_images, train_labels, epochs=30)

# Evaluación del modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)

# Predicción usando el modelo
#plt.imshow(x_test[11], cmap=plt.cm.binary)
# predictions = model.predict(x_test)
# print("Imagen 11: ", np.argmax(predictions[11]))