import tensorflow as tf

from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
for layer in model.layers:
    print(layer.name)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# # Create a TensorBoard callback
# logs = "/workspace/ml_predict/out/logs/" + "conv1"
#
# tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                  histogram_freq = 1,
#                                                  profile_batch = '500,510')
# model.fit(train_images, train_labels, epochs=2,
#                     validation_data=(test_images, test_labels),
#                     callbacks=[tboard_callback])


# from tensorflow.python.tools import freeze_graph
# from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
#
#
# # Convert Keras model to ConcreteFunction
# full_model = tf.function(lambda x: model(x))
# full_model = full_model.get_concrete_function(
#     tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name=model.input.name))
# # Get frozen ConcreteFunction
# frozen_func = convert_variables_to_constants_v2(full_model)
# frozen_func.graph.as_graph_def()
# layers = [op.name for op in frozen_func.graph.get_operations()]
# print("-" * 50)
# print("Frozen model layers: ")
# for layer in layers:
#     print(layer)
# print("-" * 50)
# print("Frozen model inputs: ")
# print(frozen_func.inputs)
# print("Frozen model outputs: ")
# print(frozen_func.outputs)
# # Save frozen graph from frozen ConcreteFunction to hard drive
# tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                   logdir="/work_space/ml_predict/out/",
#                   name="frozen_graph.pbtxt")
