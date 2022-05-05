import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
import optuna
from sklearn.model_selection import train_test_split
from .nn_model import Mlp
from .data import save_hyperparameters, get_hyperparameters, get_path_root


def train_model(X_train, y_train, tuning=True):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    n_classes = len(set(y_train))

    # split between train and eval
    x_train2, x_val, y_train2, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    if tuning:
        # Use optuna to estimate the best hyper parameters
        print('Tuning hyper parameters...')
        params = hyperparameters_tuning(x_train2, y_train2, x_val, y_val)
    else:
        # read the hyper parameters saved
        params = get_hyperparameters()
    print('Hyper parameters selected for training:')
    print(params)

    print('training the model...')
    model = training_session(X_train, y_train, n_classes, **params, epochs=300, hyper_tuning=False)
    return model


def training_session(x_train, y_train, n_classes, epochs, hyper_tuning, **params):
    model = Mlp(epochs=epochs,
                input_dim=x_train.shape[1],
                n_classes=n_classes,
                hyper_tuning=hyper_tuning,
                **params)
    model.train_model(x_train, y_train)
    return model


def hyperparameters_tuning(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array):
    """
    optimize hyperparameters
    :return:
    """
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    objective_used = get_objective(x_train, y_train, x_val, y_val)
    study.optimize(objective_used, n_trials=30)
    best_params = study.best_params
    save_hyperparameters(best_params)
    return best_params


def objective(trial: optuna.Trial, x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array):
    """
    Objective function used by Optuna.
    :param trial:
    :return:
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


def get_objective(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array):
    """
    Returns the optimization function as used by optuna given our dataset. The function returned is a function of trial.
    :param train_x: features used for training
    :param val_x: features used for validation
    :param train_y: target variable for training
    :param val_y: trarget variable for validation
    :return:
    optimization function
    """
    return lambda trial: objective(trial, x_train, y_train, x_val, y_val)


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    y_pred = model.predict(X)
    return y_pred
