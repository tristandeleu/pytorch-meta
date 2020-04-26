import pytest

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from torchmeta.modules import MetaSequential, MetaModule, MetaLinear


def test_metasequential():
    meta_model = MetaSequential(
        nn.Linear(2, 3, bias=True),
        nn.ReLU(),
        MetaLinear(3, 5, bias=True))
    model = nn.Sequential(
        nn.Linear(2, 3, bias=True),
        nn.ReLU(),
        nn.Linear(3, 5, bias=True))

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.Sequential)

    params = OrderedDict(meta_model.meta_named_parameters())
    assert set(params.keys()) == set(['2.weight', '2.bias'])

    # Set same weights for both models
    weight0 = torch.randn(3, 2)
    meta_model[0].weight.data.copy_(weight0)
    model[0].weight.data.copy_(weight0)

    bias0 = torch.randn(3)
    meta_model[0].bias.data.copy_(bias0)
    model[0].bias.data.copy_(bias0)

    weight2 = torch.randn(5, 3)
    meta_model[2].weight.data.copy_(weight2)
    model[2].weight.data.copy_(weight2)

    bias2 = torch.randn(5)
    meta_model[2].bias.data.copy_(bias2)
    model[2].bias.data.copy_(bias2)

    inputs = torch.randn(5, 2)

    outputs_torchmeta = meta_model(inputs, params=None)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


def test_metasequential_params():
    meta_model = MetaSequential(
        nn.Linear(2, 3, bias=True),
        nn.ReLU(),
        MetaLinear(3, 5, bias=True))
    model = nn.Sequential(
        nn.Linear(2, 3, bias=True),
        nn.ReLU(),
        nn.Linear(3, 5, bias=True))

    # Set same weights for both models (first layer)
    weight0 = torch.randn(3, 2)
    meta_model[0].weight.data.copy_(weight0)
    model[0].weight.data.copy_(weight0)

    bias0 = torch.randn(3)
    meta_model[0].bias.data.copy_(bias0)
    model[0].bias.data.copy_(bias0)

    params = OrderedDict()
    params['2.weight'] = torch.randn(5, 3)
    model[2].weight.data.copy_(params['2.weight'])

    params['2.bias'] = torch.randn(5)
    model[2].bias.data.copy_(params['2.bias'])

    inputs = torch.randn(5, 2)

    outputs_torchmeta = meta_model(inputs, params=params)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())
