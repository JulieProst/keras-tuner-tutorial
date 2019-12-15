from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)
from kerastuner import HyperModel


class SimpleHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(
            Dense(
                units=hp.Int('units',
                             min_value=32,
                             max_value=512,
                             step=32
                             ),
                activation='relu')
        )
        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(
            Conv2D(
                filters=32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape
            )
        )
        model.add(Conv2D(filters=64, activation='relu', kernel_size=(3, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(units=128, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
