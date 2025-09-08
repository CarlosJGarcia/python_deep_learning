# Ejemplo con TensorFlow y Keras
# Datos de entrenamiento y test numéricos

import sys
import numpy as np
import tensorflow as tf

def load_data():
    # Dummy data 
    train_num_normal = [1, 2, 3, 4, 5]
    train_label_normal = [2, 4, 6, 8, 10]
    test_num_normal = [11, 12, 13, 14, 15]
    test_label_normal = [22, 24, 26, 28, 30]

    train_num_normal = []
    train_label_normal = []
    for n in range(0,60000):
        val = np.random.randint(0, 65536)
        train_num_normal.append(val)
        train_label_normal.append(val*2)

    test_num_normal = []
    test_label_normal = []
    for n in range(0,10000):
        val = np.random.randint(0, 65536)
        test_num_normal.append(val)
        test_label_normal.append(val*2)

    # De matriz normal a matriz numpy
    train_num = np.array(train_num_normal, dtype=np.float32)
    train_label= np.array(train_label_normal, dtype=np.float32)
    test_num = np.array(test_num_normal, dtype=np.float32)
    test_label = np.array(test_label_normal, dtype=np.float32)
    
    return (train_num, train_label), (test_num, test_label)

print("Versiones:")
python_version = sys.version.split('|')[0].strip()
print("- Python", python_version)
print("- TensorFlow", tf.__version__)
print("- tf.Keras", tf.keras.__version__)

# Conjunto de entrenamiento. Una matriz de datos y otra de etiquetas
xs = np.array([1, 2, 3, 4, 5], dtype=np.float32)
ys = np.array([2, 4, 6, 8, 10], dtype=np.float32)

(xs, ys), (test_images, test_labels) = load_data()

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
# model.compile(loss="mean_squared_error", optimizer="sgd")
# model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['accuracy'])

# Entrenamiento del modelo
print("\n")
# model.fit(xs, ys, epochs=500)
model.fit(xs, ys, epochs=60)

# Make a prediction for 10
print(f"\nPrediction for 10: {model.predict(np.array([10.0]), verbose=0).item():.5f}")

# Evaluación del modelo
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_acc)