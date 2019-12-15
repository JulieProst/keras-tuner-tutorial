import time

from kerastuner.tuners import BayesianOptimization, Hyperband, RandomSearch
from tensorflow.keras.datasets import cifar10

from hypermodels import CNNHyperModel

SEED = 1

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

N_EPOCH_SEARCH = 1
HYPERBAND_MAX_EPOCHS = 2
MAX_TRIALS = 2
EXECUTION_PER_TRIAL = 1
BAYESIAN_NUM_INITIAL_POINTS = 2


def run_hyperparameter_tuning():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    tuners = define_tuners(hypermodel, directory='cifar10', project_name='simple_cnn_tuning')

    results = []
    for tuner in tuners:
        elapsed_time, loss, accuracy = tuner_evaluation(tuner, x_test, x_train, y_test, y_train)
        print(f'Elapsed time = {elapsed_time:10.6f} s, accuracy = {accuracy}, loss = {loss}')
        results.add([elapsed_time, loss, accuracy])
    print(results)


def tuner_evaluation(tuner, x_test, x_train, y_test, y_train):
    # Overview of the task
    tuner.search_space_summary()

    # Performs the hyperparameter tuning
    search_start = time.time()
    tuner.search(x_train, y_train, epochs=N_EPOCH_SEARCH, validation_split=0.1)
    search_end = time.time()
    elapsed_time = search_end - search_start

    # Show a summary of the search
    tuner.results_summary()

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]

    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(x_test, y_test)
    return elapsed_time, loss, accuracy


def define_tuners(hypermodel, directory, project_name):
    random_tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        seed=SEED,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f'{directory}_random_search',
        project_name=project_name
    )
    hyperband_tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective='val_accuracy',
        seed=SEED,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f'{directory}_hyperband',
        project_name=project_name
    )
    bayesian_tuner = BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        seed=SEED,
        num_initial_points=BAYESIAN_NUM_INITIAL_POINTS,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f'{directory}_bayesian',
        project_name=project_name
    )
    return [random_tuner, hyperband_tuner, bayesian_tuner]


if __name__ == "__main__":
    run_hyperparameter_tuning()
