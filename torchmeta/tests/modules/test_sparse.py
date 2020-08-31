import pytest

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from torchmeta.modules import MetaModule
from torchmeta.modules.sparse import MetaEmbedding, MetaEmbeddingBag


def test_metaembedding():
    meta_model = MetaEmbedding(5, 3, padding_idx=0)
    model = nn.Embedding(5, 3, padding_idx=0)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.Embedding)

    # Set same weights for both models
    weight = torch.randn(5, 3)
    meta_model.weight.data.copy_(weight)
    model.weight.data.copy_(weight)

    inputs = torch.randint(5, size=(2, 7))

    outputs_torchmeta = meta_model(inputs, params=None)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


def test_metaembedding_params():
    meta_model = MetaEmbedding(5, 3, padding_idx=0)
    model = nn.Embedding(5, 3, padding_idx=0)

    params = OrderedDict()
    params['weight'] = torch.randn(5, 3)
    model.weight.data.copy_(params['weight'])

    inputs = torch.randint(5, size=(2, 7))

    outputs_torchmeta = meta_model(inputs, params=params)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('mode', ['sum', 'mean', 'max'])
def test_metaembeddingbag(mode):
    meta_model = MetaEmbeddingBag(5, 3, mode=mode)
    model = nn.EmbeddingBag(5, 3, mode=mode)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.EmbeddingBag)

    # Set same weights for both models
    weight = torch.randn(5, 3)
    meta_model.weight.data.copy_(weight)
    model.weight.data.copy_(weight)

    inputs = torch.randint(5, size=(2, 7))

    outputs_torchmeta = meta_model(inputs, params=None)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())


@pytest.mark.parametrize('mode', ['sum', 'mean', 'max'])
def test_metaembeddingbag_params(mode):
    meta_model = MetaEmbeddingBag(5, 3, mode=mode)
    model = nn.EmbeddingBag(5, 3, mode=mode)

    params = OrderedDict()
    params['weight'] = torch.randn(5, 3)
    model.weight.data.copy_(params['weight'])

    inputs = torch.randint(5, size=(2, 7))

    outputs_torchmeta = meta_model(inputs, params=params)
    outputs_nn = model(inputs)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())
