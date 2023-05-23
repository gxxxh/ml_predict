import tensorflow as tf
# too large, need the loop in train
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
# Create a TensorBoard callback
logs = "/workspace/ml_predict/out/torch_perf/"
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tf.profiler.experimental.start(logs)
with tf.profiler.experimental.Trace("tftrace",_r=1):
    model.fit(train_images, train_labels, epochs=1,
              validation_data=(test_images, test_labels))
tf.profiler.experimental.stop()
