import pytest
import pandas as pd
from fastapi.testclient import TestClient
from main import CensusItem, app
from starter.ml.data import get_cat_features, process_data
from starter.ml.model import get_trained_mlp, inference


@pytest.fixture(scope='module')
def client():
    mock_client = TestClient(app)
    return mock_client


@pytest.fixture(scope='module')
def census_item(negative_example):
    return CensusItem(**negative_example)


def test_census_item(census_item):
    """
    Test that CensusItem is defined properly
    """
    # When creating an instance of CensusItem using a row of our data, no error should be raised
    _ = census_item


def test_api_get_root(client):
    r = client.get("/")
    assert r.status_code == 200
    output = r.json()
    assert 'greeting' in output


@pytest.fixture(scope='class')
def predict_request(client, positive_example, negative_example):
    requests = {}
    requests['positive'] = client.post("/predict/", json=positive_example)
    requests['negative'] = client.post("/predict/", json=negative_example)
    return requests


class TestAPIPredict:
    """
    Tests for API post predict
    """

    def test_api_predict_basic(self, predict_request):
        response = predict_request['negative']
        assert response.status_code == 200
        output = response.json()['predicted_salary_class']
        assert type(output) == int

    def test_api_predict(self, predict_request, positive_example, negative_example):
        """
        Test that the prediction we get from using the api is identical to the one we get from inferring
        directly from the model. We consider both an example where the model predicts a positive outcome and a case
        where the model predicts a negative outcome.
        """
        for type, example in zip(['positive', 'negative'], [positive_example, negative_example]):
            response = predict_request[type]
            output = response.json()['predicted_salary_class']
            data = pd.DataFrame([example])
            cat_features = get_cat_features()
            model = get_trained_mlp()
            x, _, _, _, _ = process_data(data, categorical_features=cat_features, label="salary",
                                         training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
            predicted = inference(model, x)
            expected_output = predicted[0]
            assert output == expected_output, f"API prediction failed for an example labelled as {type} by the model"
