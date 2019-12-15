from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def build_simple_model(hp):
    model = keras.Sequential()
    model.add(Flatten(input_shape=INPUT_SHAPE))
    model.add(
        Dense(
            units=hp.Int('units',
                         min_value=32,
                         max_value=512,
                         step=32
                         ),
            activation='relu')
    )
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn_model(hp):
    model = keras.Sequential()
    model.add(
        Conv2D(
            filters=32, kernel_size=(3, 3), activation='relu', input_shape=INPUT_SHAPE
        )
    )
    model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
