import tensorflow as tflow

from tensorflow.keras.layers import Flatten

from keras.layers.core import Dense

from tensorflow.keras.models import Sequential

from tensorflow.keras.optimizers import Adam

import pathlib


def get_data(img_height = 180, img_width=180, batch_size = 32):
    demo_dataset = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

    directory = tflow.keras.utils.get_file(
        'flower_photos', origin=demo_dataset, untar=True)

    data_directory = pathlib.Path(directory)

    train_ds = tflow.keras.preprocessing.image_dataset_from_directory(

        data_directory,

        validation_split=0.2,

        subset="training",

        seed=123,

        label_mode='categorical',

        image_size=(img_height, img_width),

        batch_size=batch_size)

    validation_ds = tflow.keras.preprocessing.image_dataset_from_directory(

        data_directory,

        validation_split=0.2,

        subset="validation",

        seed=123,

        label_mode='categorical',

        image_size=(img_height, img_width),

        batch_size=batch_size)
    return train_ds, validation_ds


train_ds, validation_ds = get_data()

demo_resnet_model = Sequential()

pretrained_model_for_demo = tflow.keras.applications.ResNet50(
    include_top=False, input_shape=(
        180, 180, 3), pooling='avg', classes=5, weights='imagenet')

demo_resnet_model.add(pretrained_model_for_demo)
demo_resnet_model.add(Flatten())

demo_resnet_model.add(Dense(512, activation='relu'))

demo_resnet_model.add(Dense(5, activation='softmax'))
for layer in demo_resnet_model.layers:
    print(layer.name)
logs = "../../../out/logs/" + "resnet"

demo_resnet_model.compile(
    optimizer=Adam(
        lr=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

print("after compile")
for layer in demo_resnet_model.layers:
    print(layer.name)

tboard_callback = tflow.keras.callbacks.TensorBoard(log_dir=logs,
                                                    histogram_freq=1,
                                                    profile_batch='1,10')
history = demo_resnet_model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=1,
    callbacks=[tboard_callback]
)
