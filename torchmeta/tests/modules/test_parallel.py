import pytest

import torch
import torch.nn as nn

from torchmeta.modules import MetaLinear, MetaSequential
from torchmeta.modules import DataParallel
from torchmeta.utils.gradient_based import gradient_update_parameters

is_multi_gpu = (torch.cuda.device_count() > 1)

def model():
    model = MetaSequential(
        MetaLinear(2, 3, bias=True),
        nn.ReLU(),
        MetaLinear(3, 1, bias=False))

    return model

def linear_model():
    return MetaLinear(2, 1)

def params(prefix=''):
    weight_0 = torch.tensor([
        [0.02, 0.03],
        [0.05, 0.07],
        [0.11, 0.13]], dtype=torch.float32)
    bias_0 = torch.tensor([0.17, 0.19, 0.23], dtype=torch.float32)
    weight_2 = torch.tensor([[0.29, 0.31, 0.37]], dtype=torch.float32)

    return ({
        f'{prefix}0.weight': weight_0,
        f'{prefix}0.bias': bias_0,
        f'{prefix}2.weight': weight_2
    })

def linear_params(prefix=''):
    weight = torch.tensor([[0.02, 0.03]], dtype=torch.float32)
    bias = torch.tensor([0.05], dtype=torch.float32)

    return {f'{prefix}weight': weight, f'{prefix}bias': bias}

def _dict_to_device(params, device):
    return dict((name, param.to(device=device))
        for (name, param) in params.items())


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
@pytest.mark.parametrize('model', [linear_model(), model()])
def test_dataparallel(model):
    device = torch.device('cuda:0')
    model = DataParallel(model)
    model.to(device=device)

    inputs = torch.rand(5, 2).to(device=device)
    outputs = model(inputs)

    assert outputs.shape == (5, 1)
    assert outputs.device == device


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
@pytest.mark.parametrize('model,params', [
    (linear_model(), linear_params('')),
    (linear_model(), linear_params('module.')),
    (model(), params('')),
    (model(), params('module.'))
])
def test_dataparallel_params(model, params):
    device = torch.device('cuda:0')
    model = DataParallel(model)
    model.to(device=device)
    params = _dict_to_device(params, device)

    inputs = torch.rand(5, 2).to(device=device)
    outputs = model(inputs, params=params)

    assert outputs.shape == (5, 1)
    assert outputs.device == device


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
@pytest.mark.parametrize('model', [linear_model(), model()])
def test_dataparallel_params_none(model):
    device = torch.device('cuda:0')
    model = DataParallel(model)
    model.to(device=device)

    inputs = torch.rand(5, 2).to(device=device)
    outputs = model(inputs, params=None)

    assert outputs.shape == (5, 1)
    assert outputs.device == device


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
@pytest.mark.parametrize('model', [linear_model(), model()])
def test_dataparallel_params_maml(model):
    device = torch.device('cuda:0')
    model = DataParallel(model)
    model.to(device=device)

    train_inputs = torch.rand(5, 2).to(device=device)
    train_outputs = model(train_inputs)

    inner_loss = train_outputs.sum()  # Dummy loss
    params = gradient_update_parameters(model, inner_loss)

    test_inputs = torch.rand(5, 2).to(device=device)
    test_outputs = model(test_inputs, params=params)

    assert test_outputs.shape == (5, 1)
    assert test_outputs.device == device

    outer_loss = test_outputs.sum()  # Dummy loss
    outer_loss.backward()
