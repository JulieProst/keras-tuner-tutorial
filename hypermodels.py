from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D


class CNNHyperModel(HyperModel):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        model = keras.Sequential()
        model.add(
            Conv2D(
                filters=16,
                kernel_size=3,
                activation="relu",
                input_shape=self.input_shape,
            )
        )
        model.add(Conv2D(filters=16, activation="relu", kernel_size=3))
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_1", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                )
            )
        )
        model.add(Conv2D(filters=32, kernel_size=3, activation="relu"))
        model.add(
            Conv2D(
                filters=hp.Choice("num_filters", values=[32, 64], default=64,),
                activation="relu",
                kernel_size=3,
            )
        )
        model.add(MaxPooling2D(pool_size=2))
        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_2", min_value=0.0, max_value=0.5, default=0.25, step=0.05,
                )
            )
        )
        model.add(Flatten())
        model.add(
            Dense(
                units=hp.Int(
                    "units", min_value=32, max_value=512, step=32, default=128
                ),
                activation=hp.Choice(
                    "dense_activation",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu",
                ),
            )
        )
        model.add(
            Dropout(
                rate=hp.Float(
                    "dropout_3", min_value=0.0, max_value=0.5, default=0.25, step=0.05
                )
            )
        )
        model.add(Dense(self.num_classes, activation="softmax"))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-4,
                    max_value=1e-2,
                    sampling="LOG",
                    default=1e-3,
                )
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
