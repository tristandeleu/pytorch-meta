import pytest

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from torchmeta.modules import MetaModule
from torchmeta.modules.linear import MetaLinear, MetaBilinear


@pytest.mark.parametrize('bias', [True, False])
def test_metalinear(bias):
    meta_model = MetaLinear(2, 3, bias=bias)
    model = nn.Linear(2, 3, bias=bias)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.Linear)

    # Set same weights for both models
    weight = torch.randn(3, 2)
    meta_model.weight.data.copy_(weight)
    model.weight.data.copy_(weight)

    if bias:
        bias = torch.randn(3)
        meta_model.bias.data.copy_(bias)
        model.bias.data.copy_(bias)

    inputs = torch.randn(5, 2)

    outputs_torchmeta = meta_model(inputs, params=None)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
def test_metalinear_params(bias):
    meta_model = MetaLinear(2, 3, bias=bias)
    model = nn.Linear(2, 3, bias=bias)

    params = OrderedDict()
    params['weight'] = torch.randn(3, 2)
    model.weight.data.copy_(params['weight'])

    if bias:
        params['bias'] = torch.randn(3)
        model.bias.data.copy_(params['bias'])

    inputs = torch.randn(5, 2)

    outputs_torchmeta = meta_model(inputs, params=params)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
def test_metabilinear(bias):
    meta_model = MetaBilinear(2, 3, 5, bias=bias)
    model = nn.Bilinear(2, 3, 5, bias=bias)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.Bilinear)

    # Set same weights for both models
    weight = torch.randn(5, 2, 3)
    meta_model.weight.data.copy_(weight)
    model.weight.data.copy_(weight)

    if bias:
        bias = torch.randn(5)
        meta_model.bias.data.copy_(bias)
        model.bias.data.copy_(bias)

    inputs1 = torch.randn(7, 2)
    inputs2 = torch.randn(7, 3)

    outputs_torchmeta = meta_model(inputs1, inputs2, params=None)
    outputs_nn = model(inputs1, inputs2)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
def test_metabilinear_params(bias):
    meta_model = MetaBilinear(2, 3, 5, bias=bias)
    model = nn.Bilinear(2, 3, 5, bias=bias)

    params = OrderedDict()
    params['weight'] = torch.randn(5, 2, 3)
    model.weight.data.copy_(params['weight'])

    if bias:
        params['bias'] = torch.randn(5)
        model.bias.data.copy_(params['bias'])

    inputs1 = torch.randn(7, 2)
    inputs2 = torch.randn(7, 3)

    outputs_torchmeta = meta_model(inputs1, inputs2, params=params)
    outputs_nn = model(inputs1, inputs2)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())
