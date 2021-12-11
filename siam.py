import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import cv2

# python (3.9)
# tensorflow (2.7.0)
# matplotlib (3.5.0)
# opencv-python (4.5.4.60)

PATCH_TREIN = 'E:\project\siam\data'
PATCH_VAL = 'E:\project\siam\set_val'
PATCH_TEST = 'E:\project\siam\set_test'

epochs = 15
batch_size = 32
margin = 1  # Наценка на конструктивный убыток.

def data_set_read(patch):
    index = 0
    index_foto = []
    foto = []
    list = os.listdir(patch)
    for i in list:
        if os.path.isdir(patch + '\\' + i):
            tmp = os.listdir(patch + '\\' + i)
            for a in tmp:
                image = cv2.imread(patch + '\\' + i + '\\' + a)
                foto.append(image[:, :, 0])
                index_foto.append(index)
            index += 1
    index_foto = np.array(index_foto)
    foto = np.array(foto)
    return foto, index_foto

trein_x, trein_y = data_set_read(PATCH_TREIN)
test_x, test_y = data_set_read(PATCH_VAL)
test_x_x, test_y_y = data_set_read(PATCH_TEST)

def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: Список, содержащий изображения, каждый индекс в этом списке соответствует одному изображению.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # добавить правельный пример
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # добавьте не правельный пример
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    if test:
        plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    else:
        plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show()

# создание тренировочной пары
pairs_train, labels_train = make_pairs(trein_x, trein_y)

# создайте пары проверки
pairs_val, labels_val = make_pairs(test_x, test_y)

# создание пары тестирования
pairs_test, labels_test = make_pairs(test_x_x, test_y_y)

x_train_1 = pairs_train[:, 0]
x_train_2 = pairs_train[:, 1]

x_val_1 = pairs_val[:, 0]
x_val_2 = pairs_val[:, 1]

x_test_1 = pairs_test[:, 0]
x_test_2 = pairs_test[:, 1]


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


input = layers.Input((100, 100, 1))
x = tf.keras.layers.BatchNormalization()(input)
x = layers.Conv2D(64, (5, 5), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation="tanh")(x)
x = layers.Conv2D(128, (3, 3), activation="tanh")(x)
x = layers.AveragePooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)

x = tf.keras.layers.BatchNormalization()(x)
x = layers.Dense(1000, activation="tanh")(x)
#x = layers.Dropout(0.25)(x)
x = layers.Dense(100, activation="tanh")(x)
embedding_network = keras.Model(input, x)


input_1 = layers.Input((100, 100, 1))
input_2 = layers.Input((100, 100, 1))

# Как упоминалось выше, сиамская сеть разделяет веса между сети вышек (дочерние сети).
# Чтобы позволить это, мы будем использовать одна и та же сеть встраивания для обеих сетей вышек.

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
#normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(merge_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

def loss(margin=1):
    """Обеспечивает 'constrastive_loss' охватывающая область с переменной 'margin'.

  Arguments:
      margin: Целое число, определяет базовую линию для расстояния, для которого пары
      следует классифицировать как непохожие. - (default is 1).

  Returns:
      'constrastive_loss' функция с данными ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: Список меток, каждая метка имеет тип float32.
          y_pred: Список прогнозов той же длины, что и у y_true,
каждая метка имеет тип float32.

      Returns:
          Тензор, содержащий конструктивные потери в виде значения с плавающей запятой.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss

siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])
siamese.summary()

visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=6)

visualize(pairs_val[:-1], labels_val[:-1], to_show=4, num_col=6)

history = siamese.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
)

def plt_metric(history, metric, title, has_valid=True):
    """Выводит заданную "метрику" из "истории'.

    Arguments:
        history: атрибут истории объекта истории, возвращенный из Model.fit.
        metric: Метрика для построения, строковое значение, присутствующее в качестве ключа в 'history'.
        title: Строка, которая будет использоваться в качестве заголовка
        has_valid: Boolean,  значение true, если в модель были переданы допустимые данные else false.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


siamese.save_weights('test_10_32_1.h5')
# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

# Plot the constrastive loss
plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

results = siamese.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)

predictions = siamese.predict([x_test_1, x_test_2])
visualize(pairs_test, labels_test, to_show=4, predictions=predictions, test=True, num_col=8)

