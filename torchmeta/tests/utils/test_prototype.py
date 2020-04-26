import pytest

import numpy as np
import torch

from torchmeta.utils.prototype import get_num_samples, get_prototypes, prototypical_loss


@pytest.mark.parametrize('dtype', [None, torch.float32])
def test_get_num_samples(dtype):
    # Numpy
    num_classes = 3
    targets_np = np.random.randint(0, num_classes, size=(2, 5))

    # PyTorch
    targets_th = torch.as_tensor(targets_np)
    num_samples_th = get_num_samples(targets_th, num_classes, dtype=dtype)

    num_samples_np = np.zeros((2, num_classes), dtype=np.int_)
    for i in range(2):
        for j in range(5):
            num_samples_np[i, targets_np[i, j]] += 1

    assert num_samples_th.shape == (2, num_classes)
    if dtype is not None:
        assert num_samples_th.dtype == dtype
    np.testing.assert_equal(num_samples_th.numpy(), num_samples_np)


def test_get_prototypes():
    # Numpy
    num_classes = 3
    embeddings_np = np.random.rand(2, 5, 7).astype(np.float32)
    targets_np = np.random.randint(0, num_classes, size=(2, 5))

    # PyTorch
    embeddings_th = torch.as_tensor(embeddings_np)
    targets_th = torch.as_tensor(targets_np)
    prototypes_th = get_prototypes(embeddings_th, targets_th, num_classes)

    assert prototypes_th.shape == (2, num_classes, 7)
    assert prototypes_th.dtype == embeddings_th.dtype

    prototypes_np = np.zeros((2, num_classes, 7), dtype=np.float32)
    num_samples_np = np.zeros((2, num_classes), dtype=np.int_)
    for i in range(2):
        for j in range(5):
            num_samples_np[i, targets_np[i, j]] += 1
            for k in range(7):
                prototypes_np[i, targets_np[i, j], k] += embeddings_np[i, j, k]

    for i in range(2):
        for j in range(num_classes):
            for k in range(7):
                prototypes_np[i, j, k] /= max(num_samples_np[i, j], 1)

    np.testing.assert_allclose(prototypes_th.detach().numpy(), prototypes_np)
