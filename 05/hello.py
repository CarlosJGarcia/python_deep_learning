# Librerías Deep Learning
import sys
import numpy as np
import tensorflow as tf


print("Versiones:")
python_version = sys.version.split('|')[0].strip()
print("- Python", python_version)
print("- TensorFlow", tf.__version__)
print("- tf.Keras", tf.keras.__version__)

# Conjunto de entrenamiento. Una matriz de datos y otra de etiquetas
xs = np.array([1, 2, 3, 4, 5], dtype=np.float32)
ys = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# Best practice - comprobar la forma de los datos
print("\nDataset check:")
print("- xs shape: ", xs.shape)
print("- xs len: ", len(xs))
print("- ys shape: ", ys.shape)
print("- ys: ", len(ys))

# Definición del modelo
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(1,)))
model.add(tf.keras.layers.Dense(units=1))

print("\n")
model.summary()
model.compile(loss="mean_squared_error", optimizer="sgd")

# Entrenamiento del modelo
print("\n")
model.fit(xs, ys, epochs=500)

# Make a prediction for 10
print(f"\nPrediction for 10: {model.predict(np.array([10.0]), verbose=0).item():.5f}")