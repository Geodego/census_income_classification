import pytest
from ..ml.data import get_raw_data


@pytest.fixture(scope='module')
def raw_data():
    """
    Get raw data and return a pd.DataFrame
    """
    df = get_raw_data()
    return df
