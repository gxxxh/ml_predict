"""
no input shape
operators without layers

"""
import tensorflow as tf

from tensorflow.keras import datasets, layers, models

from tensorflow.python.client import timeline

tf.compat.v1.disable_eager_execution()

# 需要传入run options才能返回得到run metadata
# HARDWARE_TRACE,SOFTWARE_TRACE
run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.HARDWARE_TRACE)
run_metadata = tf.compat.v1.RunMetadata()

# 这里是构建Keras模型的代码
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],
              options=run_options,
              run_metadata=run_metadata,
              )

model.fit(train_images, train_labels, epochs=1,
          validation_data=(test_images, test_labels))

# 运行后保存timeline
tl = timeline.Timeline(run_metadata.step_stats)
ctf = tl.generate_chrome_trace_format()
with open("/workspace/ml_predict/out/torch_perf/timeline2.json", 'w') as f:
    f.write(ctf)
