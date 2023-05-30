from tensorflow.keras.layers import Flatten
from keras.layers.core import Dense
from tensorflow.keras.models import Sequential
from tf_models.models.resnet50 import get_resnet50


def get_model(model_name):
    model = Sequential()
    if model_name == 'resnet50':
        model.add(get_resnet50())
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))

    model.add(Dense(5, activation='softmax'))
    return model


def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer, loss, metrics)
