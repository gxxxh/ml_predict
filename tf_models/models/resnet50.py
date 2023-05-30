import tensorflow as tf




def get_resnet50(include_top=False, input_shape=(180, 180, 3), pooling='avg', classes=5):
    """
    pooling = 'avg, max'
    """
    model = tf.keras.applications.ResNet50(
        include_top=include_top,
        weights=None,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes
    )
    return model
