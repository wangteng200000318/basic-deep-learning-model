import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv


def data_decode_resize(file_name):
    image_string = tf.io.read_file(file_name)
    image = tf.image.decode_jpeg(image_string)
    image = tf.image.resize(image, [64, 64]) / 255.0
    # plt.imshow(image)
    # plt.show()
    # print(image)
    return image


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(256, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Conv2D(512, 3, activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.sigmoid)
    ])


def data_processing(data_dir):
    train_filenames = [data_dir + file_name for file_name in sorted(os.listdir(data_dir))]

    # if the categorical is cat, value is 0 else 1
    train_label = np.array([0 if 'cat' == file_name[41:44] else 1 for file_name in train_filenames])
    print(train_label)
    train_feature = np.array([data_decode_resize(file_name) for file_name in train_filenames])

    return train_feature, train_label


def data_test(test_dir):
    test_filenames = [test_dir + file_name for file_name in sorted(os.listdir(test_dir))]
    # print(test_filenames)
    test_feature = np.array(([data_decode_resize(file_name) for file_name in test_filenames]))
    return test_feature


if __name__ == '__main__':
    model = build_model()
    model.summary()
    check_point = tf.train.Checkpoint(model=model)
    # check_point.restore(tf.train.latest_checkpoint('./save'))
    #
    train_feature, train_label = data_processing('dogs-vs-cats-redux-kernels-edition/train/')
    # print(model.predict(train_feature))
    model.compile(optimizer=tf.optimizers.Adam(), loss=tf.losses.binary_crossentropy, metrics=['accuracy'])
    model.fit(train_feature, train_label, epochs=10)
    check_point .save('./save/model.ckpt')
    # predict: 1 is dog and 0 is cat
    test_feature = data_test('dogs-vs-cats-redux-kernels-edition/test/')

    res = model.predict(test_feature)
    print(res)
    with open('dogs-vs-cats-redux-kernels-edition/sample_submission.csv', 'w')as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        for i in range(12500):
            writer.writerow([i + 1, res[i][0]])
