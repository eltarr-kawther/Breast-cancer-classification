from keras.applications.xception import Xception
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Resizing

def get_xception():
    base_model = Xception(include_top=False, input_shape=(71, 71, 3))
    base_model.trainable = False
    
    flatten_layer = Flatten()
    dense_layer = lambda x: Dense(x, activation='relu')
    prediction_layer = Dense(2, activation='softmax')
    dropout_layer = lambda x: Dropout(x)
    
    model = Sequential([
        Resizing(71, 71),
        base_model,
        flatten_layer,
        dense_layer(16),
        dropout_layer(.25),
        dense_layer(32),
        dropout_layer(.5),
        prediction_layer
    ])
    return model
