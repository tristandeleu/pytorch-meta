import pytest

import numpy as np
import torch

from torchmeta.utils.matching import (pairwise_cosine_similarity,
    matching_log_probas, matching_probas, matching_loss)


def test_pairwise_similarity():
    eps = 1e-8
    # Numpy
    embeddings1_np = np.random.randn(2, 3, 7).astype(np.float32)
    embeddings2_np = np.random.randn(2, 5, 7).astype(np.float32)

    # PyTorch
    embeddings1_th = torch.as_tensor(embeddings1_np)
    embeddings2_th = torch.as_tensor(embeddings2_np)

    similarities_th = pairwise_cosine_similarity(embeddings1_th, embeddings2_th,
                                                 eps=eps)

    assert similarities_th.shape == (2, 3, 5)

    norm1_np = np.sqrt(np.sum(embeddings1_np ** 2, axis=2))
    norm2_np = np.sqrt(np.sum(embeddings2_np ** 2, axis=2))
    similarities_np = np.zeros((2, 3, 5), dtype=np.float32)
    for i in range(2):
        for j in range(3):
            for k in range(5):
                for l in range(7):
                    similarities_np[i, j, k] += embeddings1_np[i, j, l] * embeddings2_np[i, k, l]
                similarities_np[i, j, k] /= max(norm1_np[i, j] * norm2_np[i, k], eps)
    np.testing.assert_allclose(similarities_th.numpy(), similarities_np, atol=1e-5)


def test_pairwise_similarity_zero():
    eps = 1e-8
    # Numpy
    embeddings1_np = np.random.randn(2, 3, 7).astype(np.float32)
    embeddings2_np = np.random.randn(2, 5, 7).astype(np.float32)

    embeddings1_np[0, 1] = 0.  # Zero out one embedding

    # PyTorch
    embeddings1_th = torch.as_tensor(embeddings1_np)
    embeddings2_th = torch.as_tensor(embeddings2_np)

    similarities_th = pairwise_cosine_similarity(embeddings1_th, embeddings2_th,
                                                 eps=eps)

    assert similarities_th.shape == (2, 3, 5)

    norm1_np = np.sqrt(np.sum(embeddings1_np ** 2, axis=2))
    norm2_np = np.sqrt(np.sum(embeddings2_np ** 2, axis=2))
    similarities_np = np.zeros((2, 3, 5), dtype=np.float32)
    for i in range(2):
        for j in range(3):
            for k in range(5):
                for l in range(7):
                    similarities_np[i, j, k] += embeddings1_np[i, j, l] * embeddings2_np[i, k, l]
                similarities_np[i, j, k] /= max(norm1_np[i, j] * norm2_np[i, k], eps)
    np.testing.assert_allclose(similarities_th.numpy(), similarities_np, atol=1e-5)


def test_matching_log_probas():
    num_classes = 11
    eps = 1e-8
    # Numpy
    embeddings_np = np.random.randn(2, 3, 7).astype(np.float32)
    test_embeddings_np = np.random.randn(2, 5, 7).astype(np.float32)
    targets_np = np.random.randint(num_classes, size=(2, 3))

    # PyTorch
    embeddings_th = torch.as_tensor(embeddings_np)
    test_embeddings_th = torch.as_tensor(test_embeddings_np)
    targets_th = torch.as_tensor(targets_np)

    log_probas_th = matching_log_probas(embeddings_th,
                                        targets_th,
                                        test_embeddings_th,
                                        num_classes,
                                        eps=eps)

    assert log_probas_th.shape == (2, 11, 5)
    np.testing.assert_array_less(log_probas_th.numpy(), 0.)


def test_matching_probas():
    num_classes = 11
    eps = 1e-8
    # Numpy
    embeddings_np = np.random.randn(2, 3, 7).astype(np.float32)
    test_embeddings_np = np.random.randn(2, 5, 7).astype(np.float32)
    targets_np = np.random.randint(num_classes, size=(2, 3))

    # PyTorch
    embeddings_th = torch.as_tensor(embeddings_np)
    test_embeddings_th = torch.as_tensor(test_embeddings_np)
    targets_th = torch.as_tensor(targets_np)

    probas_th = matching_probas(embeddings_th,
                                targets_th,
                                test_embeddings_th,
                                num_classes,
                                eps=eps)

    assert probas_th.shape == (2, 11, 5)
    assert np.all(probas_th.numpy() >= 0.)
    assert np.all(probas_th.numpy() <= 1.)
    np.testing.assert_allclose(probas_th.sum(1).numpy(), 1., atol=1e-5)

    similarities_th = pairwise_cosine_similarity(embeddings_th,
                                                 test_embeddings_th,
                                                 eps=eps)
    exp_similarities_np = np.exp(similarities_th.numpy())

    probas_np = np.zeros((2, 11, 5), dtype=np.float32)
    for i in range(2):
        for j in range(3):
            l = targets_np[i, j]
            for k in range(5):
                probas_np[i, l, k] += (exp_similarities_np[i, j, k]
                    / np.sum(exp_similarities_np[i, :, k]))

    np.testing.assert_allclose(probas_th.numpy(), probas_np, atol=1e-5)
