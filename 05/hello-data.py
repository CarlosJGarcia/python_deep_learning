import numpy as np

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

"""
# El primer elemento es la matriz de imagenes y el segundo la matriz de etiquetas
(train_images, train_labels), (test_images, test_labels) = load_data()

# Dataset check
print("\nDataset check:")
print("- tran_images shape: ", train_images.shape)
print("- train_labels len: ", len(train_labels))
print("- test_images shape: ", test_images.shape)
print("- test_labels len: ", len(test_labels))
"""