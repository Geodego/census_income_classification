from ..ml.data import get_path_root, get_hyperparameters, get_cat_features


def test_get_path_root():
    """
    Test that path returned is the absolut path to the project directory
    """
    path_root = get_path_root()
    assert path_root.name == 'census_income_classification'
    assert path_root.is_absolute()


def test_get_raw_data(raw_data):
    """
    Test if the raw data we get are as expected
    """
    assert not raw_data.empty
    assert len(raw_data.columns) == 15


def test_get_hyperparameters():
    params = get_hyperparameters()['parameters']
    assert set(params.keys()) == {'batch_size', 'dropout_rate', 'hidden_dim', 'learning_rate', 'n_layers'}


def test_get_cat_features():
    exp_cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    cat_features = get_cat_features()
    assert cat_features == exp_cat_features
