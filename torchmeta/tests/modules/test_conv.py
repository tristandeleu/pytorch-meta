import pytest

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from torchmeta.modules import MetaModule
from torchmeta.modules.conv import MetaConv1d, MetaConv2d, MetaConv3d


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('padding_mode', [None, 'zeros', 'reflect', 'replicate', 'circular'])
def test_metaconv1d(bias, padding_mode):
    padding = 0 if padding_mode is None else 2
    padding_mode = padding_mode or 'zeros'
    meta_model = MetaConv1d(2, 3, kernel_size=5, padding=padding,
        padding_mode=padding_mode, bias=bias)
    model = nn.Conv1d(2, 3, kernel_size=5, padding=padding,
        padding_mode=padding_mode, bias=bias)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.Conv1d)

    # Set same weights for both models
    weight = torch.randn(3, 2, 5)
    meta_model.weight.data.copy_(weight)
    model.weight.data.copy_(weight)

    if bias:
        bias = torch.randn(3)
        meta_model.bias.data.copy_(bias)
        model.bias.data.copy_(bias)

    inputs = torch.randn(7, 2, 11)

    outputs_torchmeta = meta_model(inputs, params=None)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('padding_mode', [None, 'zeros', 'reflect', 'replicate', 'circular'])
def test_metaconv1d_params(bias, padding_mode):
    padding = 0 if padding_mode is None else 2
    padding_mode = padding_mode or 'zeros'
    meta_model = MetaConv1d(2, 3, kernel_size=5, padding=padding,
        padding_mode=padding_mode, bias=bias)
    model = nn.Conv1d(2, 3, kernel_size=5, padding=padding,
        padding_mode=padding_mode, bias=bias)

    params = OrderedDict()
    params['weight'] = torch.randn(3, 2, 5)
    model.weight.data.copy_(params['weight'])

    if bias:
        params['bias'] = torch.randn(3)
        model.bias.data.copy_(params['bias'])

    inputs = torch.randn(7, 2, 11)

    outputs_torchmeta = meta_model(inputs, params=params)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('padding_mode', [None, 'zeros', 'reflect', 'replicate', 'circular'])
def test_metaconv2d(bias, padding_mode):
    padding = 0 if padding_mode is None else (2, 3)
    padding_mode = padding_mode or 'zeros'
    meta_model = MetaConv2d(2, 3, kernel_size=(5, 7), padding=padding,
        padding_mode=padding_mode, bias=bias)
    model = nn.Conv2d(2, 3, kernel_size=(5, 7), padding=padding,
        padding_mode=padding_mode, bias=bias)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.Conv2d)

    # Set same weights for both models
    weight = torch.randn(3, 2, 5, 7)
    meta_model.weight.data.copy_(weight)
    model.weight.data.copy_(weight)

    if bias:
        bias = torch.randn(3)
        meta_model.bias.data.copy_(bias)
        model.bias.data.copy_(bias)

    inputs = torch.randn(11, 2, 13, 17)

    outputs_torchmeta = meta_model(inputs, params=None)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('padding_mode', [None, 'zeros', 'reflect', 'replicate', 'circular'])
def test_metaconv2d_params(bias, padding_mode):
    padding = 0 if padding_mode is None else (2, 3)
    padding_mode = padding_mode or 'zeros'
    meta_model = MetaConv2d(2, 3, kernel_size=(5, 7), padding=padding,
        padding_mode=padding_mode, bias=bias)
    model = nn.Conv2d(2, 3, kernel_size=(5, 7), padding=padding,
        padding_mode=padding_mode, bias=bias)

    params = OrderedDict()
    params['weight'] = torch.randn(3, 2, 5, 7)
    model.weight.data.copy_(params['weight'])

    if bias:
        params['bias'] = torch.randn(3)
        model.bias.data.copy_(params['bias'])

    inputs = torch.randn(11, 2, 13, 17)

    outputs_torchmeta = meta_model(inputs, params=params)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('padding_mode', [None, 'zeros', 'replicate', 'circular'])
def test_metaconv3d(bias, padding_mode):
    padding = 0 if padding_mode is None else (2, 3, 5)
    padding_mode = padding_mode or 'zeros'
    meta_model = MetaConv3d(2, 3, kernel_size=(5, 7, 11), padding=padding,
        padding_mode=padding_mode, bias=bias)
    model = nn.Conv3d(2, 3, kernel_size=(5, 7, 11), padding=padding,
        padding_mode=padding_mode, bias=bias)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.Conv3d)

    # Set same weights for both models
    weight = torch.randn(3, 2, 5, 7, 11)
    meta_model.weight.data.copy_(weight)
    model.weight.data.copy_(weight)

    if bias:
        bias = torch.randn(3)
        meta_model.bias.data.copy_(bias)
        model.bias.data.copy_(bias)

    inputs = torch.randn(13, 2, 17, 23, 29)

    outputs_torchmeta = meta_model(inputs, params=None)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('padding_mode', [None, 'zeros', 'replicate', 'circular'])
def test_metaconv3d_params(bias, padding_mode):
    padding = 0 if padding_mode is None else (2, 3, 5)
    padding_mode = padding_mode or 'zeros'
    meta_model = MetaConv3d(2, 3, kernel_size=(5, 7, 11), padding=padding,
        padding_mode=padding_mode, bias=bias)
    model = nn.Conv3d(2, 3, kernel_size=(5, 7, 11), padding=padding,
        padding_mode=padding_mode, bias=bias)

    params = OrderedDict()
    params['weight'] = torch.randn(3, 2, 5, 7, 11)
    model.weight.data.copy_(params['weight'])

    if bias:
        params['bias'] = torch.randn(3)
        model.bias.data.copy_(params['bias'])

    inputs = torch.randn(13, 2, 17, 23, 29)

    outputs_torchmeta = meta_model(inputs, params=params)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())
