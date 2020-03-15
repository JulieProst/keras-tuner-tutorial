import time

from kerastuner.tuners import (
    BayesianOptimization,
    Hyperband,
    RandomSearch,
)
from loguru import logger
from pathlib import Path

from hypermodels import CNNHyperModel
from utils import (
    set_gpu_config,
    load_data,
)

SEED = 1

NUM_CLASSES = 10
INPUT_SHAPE = (32, 32, 3)

N_EPOCH_SEARCH = 40
HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2
BAYESIAN_NUM_INITIAL_POINTS = 1


def run_hyperparameter_tuning():
    x_test, x_train, y_test, y_train = load_data()

    hypermodel = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

    output_dir = Path("./output/cifar10/")
    tuners = define_tuners(
        hypermodel, directory=output_dir, project_name="simple_cnn_tuning"
    )

    results = []
    for tuner in tuners:
        elapsed_time, loss, accuracy = tuner_evaluation(
            tuner, x_test, x_train, y_test, y_train
        )
        logger.info(
            f"Elapsed time = {elapsed_time:10.4f} s, accuracy = {accuracy}, loss = {loss}"
        )
        results.append([elapsed_time, loss, accuracy])
    logger.info(results)


def tuner_evaluation(tuner, x_test, x_train, y_test, y_train):
    set_gpu_config()

    # Overview of the task
    tuner.search_space_summary()

    # Performs the hyperparameter tuning
    logger.info("Start hyperparameter tuning")
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
        objective="val_accuracy",
        seed=SEED,
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f"{directory}_random_search",
        project_name=project_name,
    )
    hyperband_tuner = Hyperband(
        hypermodel,
        max_epochs=HYPERBAND_MAX_EPOCHS,
        objective="val_accuracy",
        seed=SEED,
        executions_per_trial=EXECUTION_PER_TRIAL,
        directory=f"{directory}_hyperband",
        project_name=project_name,
    )
    bayesian_tuner = BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        seed=SEED,
        num_initial_points=BAYESIAN_NUM_INITIAL_POINTS,
        max_trials=MAX_TRIALS,
        directory=f"{directory}_bayesian",
        project_name=project_name
    )
    return [random_tuner, hyperband_tuner, bayesian_tuner]


if __name__ == "__main__":
    run_hyperparameter_tuning()
