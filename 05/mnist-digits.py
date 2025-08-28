# Librerías Deep Learning
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

print("Versiones:")
print("- Python", sys.version)
print("- TensorFlow", tf.__version__)
print("- tf.Keras", tf.keras.__version__)

# Dataset MNIST, que está en storage.googleapis.com en TensorFlow/Keras
mnist = tf.keras.datasets.mnist

# Defino dos tuplas, cada una con dos arrays numpy
# Train: Tupla conjunto de entrenamiento
# Test: Tupla conjunto de prueba
# En cada tupla, la primera matriz son imagenes y la segunda sus etiquetas
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocesado del dataset -> adaptar los datos para usarlos en la red neuronal
# Normalización: de uint8 [0 - 255] a float32 [0 - 1]
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

# Best practice - comprobar la forma de los datos
print("\nDataset check:")
print("- tran_images shape: ", train_images.shape)
print("- train_labels len: ", len(train_labels))
print("- test_images shape: ", test_images.shape)
print("- test_labels len: ", len(test_labels))
print("- train_labels: ", train_labels)

# Cambio de forma el tensor
# Convierto la matriz de 60000 elementos 28x28 a 60000 elemntos de 784 items
# Pasamos por tanto de 3D a 2D
train_images = train_images.reshape(60000, 784)
test_images = test_images.reshape(10000, 784)

# Cambio las etiquetas a codificación one-hot
# Los valores pasan a un vector de 10 items, cada posición es un valor
train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)



# Definición del modelo
model = Sequential()
model.add(Dense(10, activation='sigmoid', input_shape=(784,)))
model.add(Dense(10, activation='softmax'))
print("\n")
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])


# Entrenamiento del modelo
print("\n")
model.fit(train_images, train_labels, epochs=5)


# Evaluación del modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)

# Predicción usando el modelo
# plt.imshow(test_images[11], cmap=plt.cm.binary)
predictions = model.predict(test_images)
print("Imagen 11: ", np.argmax(predictions[11]))
