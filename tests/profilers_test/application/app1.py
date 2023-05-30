from tensorflow.keras.layers import Flatten
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import pathlib


def get_data(img_height=180, img_width=180, batch_size=32):
    demo_dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    directory = tf.keras.utils.get_file(
        'flower_photos', origin=demo_dataset, untar=True)

    data_directory = pathlib.Path(directory)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(

        data_directory,

        validation_split=0.2,

        subset="training",

        seed=123,

        label_mode='categorical',

        image_size=(img_height, img_width),

        batch_size=batch_size)

    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(

        data_directory,

        validation_split=0.2,

        subset="validation",

        seed=123,

        label_mode='categorical',

        image_size=(img_height, img_width),

        batch_size=batch_size)
    return train_ds, validation_ds


# create the base pre-trained model
base_model = tf.keras.applications.VGG16(include_top=False,
                                         input_shape=(180, 180, 3), pooling='avg', classes=5, weights=None)
train_ds, validation_ds = get_data()

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation='softmax'))

logs = "/workspace/ml_predict/out/logs/" + "vgg16"
tboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logs,
                                                 histogram_freq=1,
                                                 profile_batch='1,10')
# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# train the model on the new tf_models for a few epochs
model.fit(train_ds, epochs=1, callbacks=[tboard_callback])


for layer in model.layers:
    print(layer.name)
