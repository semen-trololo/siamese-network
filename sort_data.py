import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import cv2

PATH_DATA = 'E:\project\siam\set_s0'
WORK_PATH = 'E:\project\siam'

def get_foto(patch):
    """ Принимает путь (patch) к файлам.
    Возвращаем массив фотографий (100х100х1)
    """
    foto = []
    list = os.listdir(patch)
    for i in list:
        image = cv2.imread(patch + '/' + i)
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

def network (foto_x, foto_y):
    """Принимает два массива х- образец, y - распозноваемые.
    Возвращает предсказание нейросети"""
    pair = []
    pair = [[foto_x, foto_y]]
    pair = np.array(pair)
    input = layers.Input((100, 100, 1))
    x = tf.keras.layers.BatchNormalization()(input)
    x = layers.Conv2D(64, (5, 5), activation="tanh")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="tanh")(x)
    x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)

    x = tf.keras.layers.BatchNormalization()(x)
    x = layers.Dense(1000, activation="tanh")(x)
    # x = layers.Dropout(0.25)(x)
    x = layers.Dense(100, activation="tanh")(x)
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

    siamese.load_weights('test_10_16.h5')
    predictions = siamese.predict([pair[:, 0], pair[:, 1]])
    print(predictions)

data = get_foto(PATH_DATA)
network(data[0], data[1])

