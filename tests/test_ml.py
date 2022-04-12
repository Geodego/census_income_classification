import pytest
from starter.ml.data import get_raw_data, get_path_root


@pytest.fixture(scope='module')
def raw_data():
    """
    Get raw data and return a pd.DataFrame
    """
    df = get_raw_data()
    return df


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
    col_names = ['age', 'workclass', 'education']  # names that we expect to find in column headers
    assert not raw_data.empty
    assert len(raw_data.columns) == 15
