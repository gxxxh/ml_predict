# only work with train,
# don't know how to turn loss and backward into graph
from keras.applications.inception_resnet_v2 import InceptionResNetV2

import tensorflow as tf

import numpy as np


def get_detailed_stats():
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            new_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
            print(new_model.summary())
            run_meta = tf.compat.v1.RunMetadata()
            input_details = new_model.get_config()
            input_shape = input_details['layers'][0]['config']['batch_input_shape']

            _ = session.run(new_model.output, {
                new_model.input.name: np.random.normal(size=(1, input_shape[1], input_shape[2], input_shape[3]))},
                            options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
                            run_metadata=run_meta)
            # '''
            ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder
            opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
                                        ).with_step(0).with_timeline_output('test.json').build()

            res1 = tf.compat.v1.profiler.profile(
                tf.compat.v1.get_default_graph(),
                run_meta=run_meta,
                cmd='code',
                options=opts)
            print(res1)
            # '''
            # Print to stdout an analysis of the memory usage and the timing information
            # broken down by operation types.
            json_export = tf.compat.v1.profiler.profile(
                tf.compat.v1.get_default_graph(),
                run_meta=run_meta,
                cmd='op',
                options=tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory())
            text_file = open("profiler.json", "w")
            text_file.write(str(json_export))
            text_file.close()
    tf.compat.v1.reset_default_graph()


get_detailed_stats()
