"""
Model training and performance evaluation functions. Include also methods for hyperparameters search.

author: Geoffroy de Gournay
date: April 27, 2022
"""

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
import optuna
from sklearn.model_selection import train_test_split
from .nn_model import Mlp
from .data import save_hyperparameters, get_hyperparameters
from typing import Callable, Tuple


def train_model(x_train: np.array, y_train: np.array, tuning: bool = True, random_state: int = 42,
                use_saved_model: bool = False) -> Mlp:
    """
    Trains a machine learning model and returns it.
    :param x_train: Training features
    :param y_train: Training labels
    :param tuning: indicates if optuna will be used for hyperparameters tuning.
    :param random_state: random seed used for splitting data
    :param use_saved_model: indicates if the model already trained will be used as a starting point for training
    :return: Trained machine learning model.
    """

    # If we use the saved model, we use the hyperparameters saved in the yaml file
    if use_saved_model:
        assert tuning is False
    n_classes = len(set(y_train))

    # split between training and eval
    x_train2, x_val, y_train2, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    if tuning:
        # Use optuna to estimate the best hyper parameters
        print('Tuning hyper parameters...')
        params = hyperparameters_tuning(x_train2, y_train2, x_val, y_val, random_state)
    else:
        # read the hyper parameters saved
        params = get_hyperparameters()['parameters']
    print('Hyper parameters selected for training:')
    print(params)

    print('training the model...')
    model = training_session(x_train, y_train, n_classes, **params, epochs=300, hyper_tuning=False,
                             use_saved_model=use_saved_model)
    return model


def training_session(x_train: np.array, y_train: np.array, n_classes: int, epochs: int, hyper_tuning: bool,
                     use_saved_model: bool, **params) -> Mlp:
    """
    Trains the model for a given set of hyperparameters given in params.
    :param x_train: features used for training
    :param y_train: labels used for training
    :param n_classes: number of different labels
    :param epochs: number of epochs used to train the model
    :param hyper_tuning: This boolean indicates if the method is used for hyperparameters optimization. In that case the
    model is set up so that it doesn't print losses during training
    :param use_saved_model: indicates if the model already trained will be used as a starting point for training
    :param params: dictionary with hyperparameters which keys are 'batch_size', 'dropout_rate', 'hidden_dim',
    'learning_rate' and 'n_layers'.
    :return: Trained model.
    """
    model = Mlp(epochs=epochs,
                input_dim=x_train.shape[1],
                n_classes=n_classes,
                hyper_tuning=hyper_tuning,
                **params)
    if use_saved_model:
        model.load_model()
    model.train_model(x_train, y_train)
    return model


def hyperparameters_tuning(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array,
                           random_state: int) -> dict:
    """
    Select optimal hyperparameters using optuna.
    :param x_train: features used for training
    :param y_train: labels for training
    :param x_val: features used for validation
    :param y_val: labels for validation
    :param random_state: random seed used for splitting data
    :return:
    Dictionary with selected hyperparameters
    """
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    objective_used = get_objective(x_train, y_train, x_val, y_val)
    study.optimize(objective_used, n_trials=30)
    best_params = study.best_params
    save_hyperparameters(best_params, random_state)
    return best_params


def objective(trial: optuna.Trial, x_train: np.array, y_train: np.array, x_val: np.array,
              y_val: np.array) -> float:
    """
    Objective function used by Optuna. The criteria used is the f1 score
    :param trial: Optuna Trial object
    :param x_train: features used for training
    :param y_train: labels for training
    :param x_val: features used for validation
    :param y_val: labels for validation
    :return: f1 score
    """
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [5, 10, 25, 50]),
        'n_layers': trial.suggest_categorical('n_layers', [1, 2, 3, 4, 5]),
        'dropout_rate': trial.suggest_categorical('dropout_rate', [0.3, 0.4, 0.5, 0.6, 0.7])
    }
    model = training_session(x_train, y_train, **params, epochs=100, n_classes=2, hyper_tuning=True)
    preds = model.predict(x_val)
    f1 = fbeta_score(y_val, preds, beta=1, zero_division=1)
    return f1


def get_objective(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array) -> Callable:
    """
    Returns the optimization function as used by optuna given our dataset. The function returned is a function of trial.
    :param x_train: features used for training
    :param y_train: labels for training
    :param x_val: features used for validation
    :param y_val: labels for validation
    :return:
    optimization function
    """
    return lambda trial: objective(trial, x_train, y_train, x_val, y_val)


def compute_model_metrics(y: np.array, preds: np.array) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.
    :param y: Known labels, binarized.
    :param preds: Predicted labels, binarized.
    :return: tuple (precision, recall, F1)
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: Mlp, x: np.array) -> np.array:
    """
    Run model inferences and return the predictions.
    :param model: Trained machine learning model.
    :param x: Data used for prediction.
    :return: Predictions from the model
    """

    y_pred = model.predict(x)
    return y_pred


def get_trained_mlp() -> Mlp:
    """
    Return trained model for inference
    :return: Mlp model

    """
    model = Mlp(use_saved_hyper_params=True)
    model.load_model()
    return model
