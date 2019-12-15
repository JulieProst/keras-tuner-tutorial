import time

from kerastuner.tuners import RandomSearch
from tensorflow.keras.datasets import cifar10

from hypermodels import CNNHyperModel

N_EPOCH_SEARCH = 10
MAX_TRIALS = 2
EXECUTION_PER_TRIAL = 1
SEED = 1

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)


def run_hyperparameter_tuning():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        seed=SEED,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory='cifar10_random_search',
        project_name='simple_cnn'
    )

    # Overview of the task
    tuner.search_space_summary()

    # Performs the hypertuning.
    search_start = time.time()
    tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)
    search_end = time.time()
    print(f'Elapsed time = {search_end - search_start:10.6f} s')

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
