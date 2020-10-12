import pytest

import torch
import torch.nn as nn

from torchmeta.modules import MetaLinear, MetaSequential
from torchmeta.modules import DataParallel
from torchmeta.utils.gradient_based import gradient_update_parameters

is_multi_gpu = (torch.cuda.device_count() > 1)

@pytest.fixture
def model():
    model = MetaSequential(
        MetaLinear(2, 3, bias=True),
        nn.ReLU(),
        MetaLinear(3, 1, bias=False))

    return model

@pytest.fixture
def params():
    device = torch.device('cuda:0')
    weight_0 = torch.tensor([
        [0.02, 0.03],
        [0.05, 0.07],
        [0.11, 0.13]], device=device, dtype=torch.float32)
    bias_0 = torch.tensor([0.17, 0.19, 0.23],
                          device=device, dtype=torch.float32)
    weight_2 = torch.tensor([[0.29, 0.31, 0.37]],
                            device=device, dtype=torch.float32)

    return {'0.weight': weight_0, '0.bias': bias_0, '2.weight': weight_2}


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
def test_dataparallel(model):
    device = torch.device('cuda:0')
    model = DataParallel(model)
    model.to(device=device)

    inputs = torch.rand(5, 2).to(device=device)
    outputs = model(inputs)

    assert outputs.shape == (5, 1)
    assert outputs.device == device


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
def test_dataparallel_params(model, params):
    device = torch.device('cuda:0')
    model = DataParallel(model)
    model.to(device=device)

    inputs = torch.rand(5, 2).to(device=device)
    outputs = model(inputs, params=params)

    assert outputs.shape == (5, 1)
    assert outputs.device == device


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
def test_dataparallel_params_none(model):
    device = torch.device('cuda:0')
    model = DataParallel(model)
    model.to(device=device)

    inputs = torch.rand(5, 2).to(device=device)
    outputs = model(inputs, params=None)

    assert outputs.shape == (5, 1)
    assert outputs.device == device


@pytest.mark.skipif(not is_multi_gpu, reason='Requires Multi-GPU support')
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
