import os
import pathlib
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import csv


def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def get_path_root() -> pathlib.PosixPath:
    """
    :return:
    absolut path to project directory "census_income_classification"
    """
    current_path = Path(os.path.realpath(__file__)).resolve()
    path = current_path
    while path.name != 'census_income_classification':
        path = path.parent
    return path


def get_path_file(file_local_path):
    """
    Return the full path of a file given its local path
    :param file_local_path: local path of the file in the project (ex: "data/census.csv")
    """
    project_dir = get_path_root()
    raw_path = PurePath.joinpath(project_dir, file_local_path)
    return raw_path


def get_raw_data():
    """
    Get the raw data as a DataFrame
    :return:
    pd.DataFrame containing raw data as read from the csv file
    """
    raw_path = get_path_file("data/census.csv")
    raw_data = pd.read_csv(raw_path)
    return raw_data


def save_clean_data():
    """
    Remove white spaces from "census.csv" and save data processed as such to "census_clean.csv".
    """
    raw_path = get_path_file("data/census.csv")
    clean_path = get_path_file("data/census_clean.csv")
    # todo: finish here
    with open(raw_path, 'r') as f_raw, open(clean_path, 'w') as f_clean:
        reader = csv.reader(f_raw, skipinitialspace=False, delimiter=',', quoting=csv.QUOTE_NONE)
        writer = csv.writer(f_clean)
        for row in reader:
            clean_row = [item.strip() for item in row]
            writer.writerow(clean_row)


if __name__ == '__main__':
    save_clean_data()
    get_path_root()
    df = get_raw_data()
    print(df.head())
