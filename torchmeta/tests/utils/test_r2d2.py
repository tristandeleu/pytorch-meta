import pytest

import numpy as np
import torch
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules import MetaLinear
from torchmeta.utils.r2d2 import ridge_regression


@pytest.mark.parametrize('reg_lambda', [0.1, 1.])
@pytest.mark.parametrize('use_woodbury', [None, True, False])
@pytest.mark.parametrize('scale', [True, False])
@pytest.mark.parametrize('bias', [True, False])
def test_ridge_regression(reg_lambda, use_woodbury, scale, bias):
    # Numpy
    num_classes = 3
    embeddings_np = np.random.randn(5, 7).astype(np.float32)
    targets_np = np.random.randint(0, num_classes, size=(5,))

    # PyTorch
    embeddings_th = torch.as_tensor(embeddings_np)
    targets_th = torch.as_tensor(targets_np)
    model = MetaLinear(7, 3, bias=bias)

    solution = ridge_regression(embeddings_th,
                                targets_th,
                                reg_lambda,
                                num_classes,
                                use_woodbury=use_woodbury,
                                scale=scale,
                                bias=bias)

    assert solution.weight.shape == (3, 7)
    if bias:
        assert solution.bias is not None
        assert solution.bias.shape == (3,)
    else:
        assert solution.bias is None

    # Optimality criterion
    # Check if the gradient of the L2-regularized MSE at the solution
    # is close to 0
    params = OrderedDict([('weight', solution.weight.requires_grad_())])
    if bias:
        params['bias'] = solution.bias.requires_grad_()

    logits = model(embeddings_th, params=params)
    targets_binary = F.one_hot(targets_th, num_classes=num_classes).float()

    # Least-square
    loss = F.mse_loss(logits, targets_binary, reduction='sum')
    if scale:
        loss /= embeddings_th.size(0)

    # L2-regularization
    loss += reg_lambda * torch.sum(solution.weight ** 2)
    if bias:
        loss += reg_lambda * torch.sum(solution.bias ** 2)
    loss.backward()

    np.testing.assert_allclose(solution.weight.grad.numpy(), 0., atol=1e-4)
    if bias:
        np.testing.assert_allclose(solution.bias.grad.numpy(), 0., atol=1e-4)


@pytest.mark.parametrize('reg_lambda', [0.1, 1.])
@pytest.mark.parametrize('use_woodbury', [None, True, False])
@pytest.mark.parametrize('scale', [True, False])
@pytest.mark.parametrize('bias', [True, False])
def test_ridge_regression_requires_grad(reg_lambda, use_woodbury, scale, bias):
    # Numpy
    num_classes = 3
    embeddings_np = np.random.randn(5, 7).astype(np.float32)
    targets_np = np.random.randint(0, num_classes, size=(5,))

    # PyTorch
    embeddings_th = torch.as_tensor(embeddings_np).requires_grad_()
    targets_th = torch.as_tensor(targets_np)
    model = MetaLinear(7, 3, bias=bias)

    solution = ridge_regression(embeddings_th,
                                targets_th,
                                reg_lambda,
                                num_classes,
                                use_woodbury=use_woodbury,
                                scale=scale,
                                bias=bias)
    params = OrderedDict([('weight', solution.weight)])
    if bias:
        params['bias'] = solution.bias

    # Compute loss on test/query samples
    test_embeddings = torch.randn(11, 7)
    test_logits = model(test_embeddings, params=params)
    test_targets = torch.randint(num_classes, size=(11,))
    loss = F.cross_entropy(test_logits, test_targets)

    # Backpropagation
    loss.backward()

    assert embeddings_th.grad is not None


@pytest.mark.parametrize('use_woodbury', [None, True, False])
@pytest.mark.parametrize('scale', [True, False])
@pytest.mark.parametrize('bias', [True, False])
def test_ridge_regression_regression_task(use_woodbury, scale, bias):
    # Numpy
    reg_lambda = 1.
    embeddings_np = np.random.randn(5, 7).astype(np.float32)
    targets_np = np.random.randn(5, 3).astype(np.float32)

    # PyTorch
    embeddings_th = torch.as_tensor(embeddings_np)
    targets_th = torch.as_tensor(targets_np)
    model = MetaLinear(7, 3, bias=bias)

    solution = ridge_regression(embeddings_th,
                                targets_th,
                                reg_lambda,
                                num_classes=None,
                                use_woodbury=use_woodbury,
                                scale=scale,
                                bias=bias)

    assert solution.weight.shape == (3, 7)
    if bias:
        assert solution.bias.shape == (3,)

    # Optimality criterion
    # Check if the gradient of the L2-regularized MSE at the solution
    # is close to 0
    params = OrderedDict([('weight', solution.weight.requires_grad_())])
    if bias:
        params['bias'] = solution.bias.requires_grad_()

    logits = model(embeddings_th, params=params)

    # Least-square
    loss = F.mse_loss(logits, targets_th, reduction='sum')
    if scale:
        loss /= embeddings_th.size(0)

    # L2-regularization
    loss += reg_lambda * torch.sum(solution.weight ** 2)
    if bias:
        loss += reg_lambda * torch.sum(solution.bias ** 2)
    loss.backward()

    np.testing.assert_allclose(solution.weight.grad.numpy(), 0., atol=1e-4)
    if bias:
        np.testing.assert_allclose(solution.bias.grad.numpy(), 0., atol=1e-4)
