import pytest
from ..ml.data import get_raw_data, get_clean_data, get_cat_features, process_data
from ..ml.model import get_trained_mlp


@pytest.fixture(scope='session')
def raw_data():
    """
    Get raw data and return a pd.DataFrame
    """
    df = get_raw_data()
    return df


@pytest.fixture(scope='session')
def clean_data():
    """
    Get clean data and return a pd.DataFrame
    """
    df = get_clean_data()
    return df


@pytest.fixture(scope='session')
def get_examples():
    """
    This fixture creates two dictionaries of features, one that is predicted as positive and one
    that is predicted as negative  by our trained model
    """
    model = get_trained_mlp()
    df = get_clean_data()
    cat_features = get_cat_features()
    x, _, _, _, _ = process_data(df, categorical_features=cat_features, label="salary",
                                 training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
    predicted = model.predict(x)  # this is mainly to get a properly shaped dataframe
    positive = df[predicted == 1].iloc[0].to_dict()
    negative = df[predicted == 0].iloc[0].to_dict()
    return positive, negative


@pytest.fixture(scope='session')
def negative_example(get_examples):
    """
    Get an example of the data predicted as negative by the model.
    """
    example = get_examples[1]
    return example


@pytest.fixture(scope='session')
def positive_example(get_examples):
    """
    Get an example of the data predicted as positive by the model.
    """
    example = get_examples[0]
    return example
