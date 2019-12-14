from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from kerastuner.tuners import RandomSearch


NUM_CLASSES = 10


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    model.add(layers.Dense(units=hp.Int('units',
                                        min_value=32,
                                        max_value=512,
                                        step=32),
                           activation='relu'))
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model


def run_hyperparameter_tuning():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=1,
        directory='mnist_random_search',
        project_name='helloworld')

    # Overview of the task
    tuner.search_space_summary()

    # Performs the hypertuning.
    tuner.search(x_train, y_train, epochs=10, validation_split=0.1)

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(x_test, y_test)
    print('loss:', loss)
    print('accuracy:', accuracy)


if __name__ == "__main__":
    run_hyperparameter_tuning()
