import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import cv2

PATH_DATA = 'c:\project\siam\set_s0\\'
WORK_PATH = 'c:\project\siam\\'

def get_foto(patch):
    """ Принимает путь (patch) к файлам.
    Возвращаем массив фотографий (100х100х1)
    """
    foto = []
    list = os.listdir(patch)
    for i in list:
        image = cv2.imread(patch + i)
        foto.append(image[:, :, 0])
    foto = np.array(foto)
    return foto

# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Найти евклидово расстояние между двумя векторами.

    Arguments:
        vects: Список, содержащий два тензора одинаковой длины.

    Returns:
        Тензор, содержащий евклидово расстояние
        (как значение с плавающей запятой) между векторами.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))

def network ():
    """Собирает сеть, загружает веса.
    Возвращает сеть."""

    input = layers.Input((100, 100, 1))
    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(8, (5, 5), activation="tanh")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation="tanh")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    #x = layers.Dense(1000, activation="tanh")(x)
    # x = layers.Dropout(0.25)(x)
    x = layers.Dense(10, activation="tanh")(x)
    embedding_network = keras.Model(input, x)

    input_1 = layers.Input((100, 100, 1))
    input_2 = layers.Input((100, 100, 1))

    # Как упоминалось выше, сиамская сеть разделяет веса между сети вышек (дочерние сети).
    # Чтобы позволить это, мы будем использовать одна и та же сеть встраивания для обеих сетей вышек.

    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    # normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="sigmoid")(merge_layer)
    siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

    siamese.load_weights('test_2_32.h5')
    return siamese

def sort_list(data):
    index_del = []
    index_del.append(0)
    new_data = []
    new_dir = WORK_PATH + str(random.randint(1, 1000000))
    os.mkdir(new_dir)
    print(len(data))
    cv2.imwrite(new_dir + '\\' + str(0) + '.jpg', data[0])
    for i in range(1, len(data)):
        pair = [[data[0], data[i]]]
        pair = np.array(pair)
        predictions = siamese.predict([pair[:, 0], pair[:, 1]])
        if predictions[0][0] > 0.75:
            index_del.append(i)
            cv2.imwrite(new_dir + '\\' + str(i) + '.jpg', data[i])
            print(i)
    for i in range(0, len(data)):
        if i not in index_del:
            new_data.append(data[i])
    return new_data
data = get_foto(PATH_DATA)
siamese = network()
while len(data) > 2:
    data = sort_list(data)

