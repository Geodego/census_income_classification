# Script to trained_and_save_model machine learning model.
import pandas as pd
import numpy as np
from typing import Union
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from .ml.data import process_data, get_clean_data, get_cat_features, get_data_slices, get_processed_test_data
from .ml.model import train_model, compute_model_metrics, inference
from .ml.nn_model import Mlp


def trained_and_save_model(tuning=True, random_state=42, use_saved_model=False):
    """
    Train and save a model. The tols used for processing the data is saved as well, both the data processing tools
    and the model
    are saved in the 'model' folder.
    :param tuning: Set to true if hyperparameters are to be optimised. If false hyperparameters are loaded from a
    yaml file.
    :param random_state: Controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output.
    :return:
    """

    # Add code to load in the data.
    data = get_clean_data()

    # Optional enhancement, use K-fold cross validation instead of a trained_and_save_model-test split.
    train, test = train_test_split(data, test_size=0.20, random_state=random_state)
    cat_features = get_cat_features()
    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data with the process_data function.
    X_test, y_test, _, _, _ = process_data(test, categorical_features=cat_features, label="salary",
                                           training=False, encoder=encoder, lb=lb, scaler=scaler)

    # Train and save a model.
    model = train_model(X_train, y_train, tuning, random_state, use_saved_model)
    model.save_model(encoder, lb, scaler)

    y_pred = inference(model, X_test)
    # precision, recall, and F1
    evaluation = compute_model_metrics(y_test, y_pred)

    return evaluation


def model_metrics_slices(model: Mlp, selected_feature: str) -> pd.DataFrame:
    """
    Computes performance on model slices for selected_feature. The print output is saved in screenshots/slice_output.txt
    :param model: trained model which performance is measured on slices of data.
    :param selected_feature: categorical feature used for slicing
    :return:
    Dataframe whith the different values of selected_feature as rows and columns precision, recall and f1 score.
    """
    slices = get_data_slices(selected_feature, model.encoder, model.lb, model.scaler)
    slices_metrics = {}
    for key, data_dict in slices.items():
        x = data_dict['x']
        y = data_dict['y']
        slices_metrics[key] = model_metrics(model, x, y)
    slices_metrics_df = pd.DataFrame(slices_metrics).T
    with open('screenshots/slice_output.txt', 'w') as f:
        print(f'Performance on model slices for {selected_feature}:\n\n', slices_metrics_df, file=f)

    return slices_metrics_df


def model_metrics(model: Mlp, x_test: np.array = None, y_test: np.array = None) -> dict:
    """
    Compute the performance of model on test data x_test, y_test
    :param model: trained model which performance is measured
    :param x_test: test data features
    :param y_test: test data labels
    :return:
    a dictionary with the precision, recall and f1 metrics calculated on test data
    """
    if x_test is None or y_test is None:
        x_test, y_test = get_processed_test_data(model.encoder, model.lb, model.scaler)
    y_pred = inference(model, x_test)
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)
    evaluation = {'precision': precision, 'recall': recall, 'f1': f1}
    return evaluation
