from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import SeparableConv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import numpy as np

def get_cnn():
    _model = Sequential()
    _model.add(SeparableConv2D(32, (3, 3), activation="relu"))
    _model.add(BatchNormalization(axis=-1))
    _model.add(MaxPooling2D((2, 2)))

    _model.add(SeparableConv2D(32, (3, 3), activation="relu"))
    _model.add(BatchNormalization(axis=-1))
    _model.add(MaxPooling2D((2, 2)))

    _model.add(SeparableConv2D(64, (3, 3), activation="relu"))
    _model.add(BatchNormalization(axis=-1))
    _model.add(MaxPooling2D(pool_size=(2, 2)))

    _model.add(Flatten())

    _model.add(Dense(64, activation="relu"))
    _model.add(BatchNormalization(axis=-1))
    _model.add(Dropout(0.5))
    _model.add(Dense(2, activation="softmax"))
    return _model