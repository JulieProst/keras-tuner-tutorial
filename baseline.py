import time

from loguru import logger

from tuner_comparison import (
    INPUT_SHAPE,
    NUM_CLASSES,
    N_EPOCH_SEARCH,
)
from utils import (
    set_gpu_config,
    load_data,
)


def base_experiment():
    from tensorflow import keras
    from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

    set_gpu_config()
    x_test, x_train, y_test, y_train = load_data()

    model = keras.Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=INPUT_SHAPE)
    )
    model.add(Conv2D(16, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(32, 3, activation="relu"))
    model.add(Conv2D(64, 3, activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=128, activation="relu"))
    model.add(Dropout(rate=0.25))
    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Start training")
    search_start = time.time()
    model.fit(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)
    search_end = time.time()
    elapsed_time = search_end - search_start
    logger.info(f"Elapsed time (s): {elapsed_time}")

    loss, accuracy = model.evaluate(x_test, y_test)
    logger.info(f"loss: {loss}, accuracy: {accuracy}")


if __name__ == "__main__":
    base_experiment()
