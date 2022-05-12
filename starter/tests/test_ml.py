import pytest
import numpy as np
from torch import nn
from ..ml.nn_model import build_mlp
from ..ml.nn_model import Mlp


@pytest.mark.parametrize('n_layers', [1, 2])
def test_build_mlp(n_layers):
    """
    Test if the neural network is built properly
    """
    input_size = 10
    output_size = 2
    hidden_size = 5
    model = build_mlp(input_size, output_size, n_layers, hidden_size, dropout_rate=0.5)
    assert len(model) == 3 * (n_layers+1)

    assert type(model[0]) is nn.Linear
    assert model[0].in_features == input_size
    assert model[0].out_features == hidden_size
    assert type(model[1]) == nn.ReLU
    assert type(model[3]) == nn.Dropout
    assert model[-1].out_features == output_size


def test_mlp_inference():
    """
    Test if the model returns expected output when used for inference
    """
    model = Mlp(n_layers=2, hidden_dim=5,n_classes=2, input_dim=10)
    n_examples =300
    data = np.random.rand(n_examples, 10)
    output = model.predict(data)
    assert output.shape[0] == n_examples