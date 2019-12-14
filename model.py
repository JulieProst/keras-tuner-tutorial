from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10
INPUT_SHAPE = (28, 28)


def build_simple_mnist_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=INPUT_SHAPE))
    model.add(
        layers.Dense(
            units=hp.Int('units',
                         min_value=32,
                         max_value=512,
                         step=32
                         ),
            activation='relu')
    )
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model
