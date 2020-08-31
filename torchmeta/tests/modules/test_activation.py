import pytest

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from torchmeta.modules import MetaModule
from torchmeta.modules.activation import MetaMultiheadAttention


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('add_bias_kv', [True, False])
@pytest.mark.parametrize('kdim,vdim', [(None, None), (7, 11)])
def test_metamultiheadattention(bias, add_bias_kv, kdim, vdim):
    meta_model = MetaMultiheadAttention(3 * 5, 3,
                                        bias=bias,
                                        add_bias_kv=add_bias_kv,
                                        kdim=kdim, vdim=vdim)
    model = nn.MultiheadAttention(3 * 5, 3,
                                  bias=bias,
                                  add_bias_kv=add_bias_kv,
                                  kdim=kdim, vdim=vdim)

    assert isinstance(meta_model, MetaModule)
    assert isinstance(meta_model, nn.MultiheadAttention)

    # Set same weights for both models
    if not meta_model._qkv_same_embed_dim:
        q_proj_weight = torch.randn(3 * 5, 3 * 5)
        meta_model.q_proj_weight.data.copy_(q_proj_weight)
        model.q_proj_weight.data.copy_(q_proj_weight)

        k_proj_weight = torch.randn(3 * 5, meta_model.kdim)
        meta_model.k_proj_weight.data.copy_(k_proj_weight)
        model.k_proj_weight.data.copy_(k_proj_weight)

        v_proj_weight = torch.randn(3 * 5, meta_model.vdim)
        meta_model.v_proj_weight.data.copy_(v_proj_weight)
        model.v_proj_weight.data.copy_(v_proj_weight)

    else:
        in_proj_weight = torch.randn(3 * 3 * 5, 3 * 5)
        meta_model.in_proj_weight.data.copy_(in_proj_weight)
        model.in_proj_weight.data.copy_(in_proj_weight)

    if bias:
        in_proj_bias = torch.randn(3 * 3 * 5)
        meta_model.in_proj_bias.data.copy_(in_proj_bias)
        model.in_proj_bias.data.copy_(in_proj_bias)

    if add_bias_kv:
        bias_k = torch.randn(1, 1, 3 * 5)
        meta_model.bias_k.data.copy_(bias_k)
        model.bias_k.data.copy_(bias_k)

        bias_v = torch.randn(1, 1, 3 * 5)
        meta_model.bias_v.data.copy_(bias_v)
        model.bias_v.data.copy_(bias_v)

    out_proj_weight = torch.randn(3 * 5, 3 * 5)
    meta_model.out_proj.weight.data.copy_(out_proj_weight)
    model.out_proj.weight.data.copy_(out_proj_weight)

    out_proj_bias = torch.randn(3 * 5)
    meta_model.out_proj.bias.data.copy_(out_proj_bias)
    model.out_proj.bias.data.copy_(out_proj_bias)

    query = torch.randn(13, 17, 3 * 5)
    key = torch.randn(19, 17, 3 * 5 if (kdim is None) else kdim)
    value = torch.randn(19, 17, 3 * 5 if (vdim is None) else vdim)

    outputs_torchmeta, weights_torchmeta = meta_model(query, key, value, params=None)
    outputs_nn, weights_nn = model(query, key, value)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())
    np.testing.assert_equal(weights_torchmeta.detach().numpy(),
                            weights_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('add_bias_kv', [True, False])
@pytest.mark.parametrize('kdim,vdim', [(None, None), (7, 11)])
def test_metamultiheadattention_params(bias, add_bias_kv, kdim, vdim):
    meta_model = MetaMultiheadAttention(3 * 5, 3,
                                        bias=bias,
                                        add_bias_kv=add_bias_kv,
                                        kdim=kdim, vdim=vdim)
    model = nn.MultiheadAttention(3 * 5, 3,
                                  bias=bias,
                                  add_bias_kv=add_bias_kv,
                                  kdim=kdim, vdim=vdim)

    params = OrderedDict()
    if not meta_model._qkv_same_embed_dim:
        params['q_proj_weight'] = torch.randn(3 * 5, 3 * 5)
        model.q_proj_weight.data.copy_(params['q_proj_weight'])

        params['k_proj_weight'] = torch.randn(3 * 5, meta_model.kdim)
        model.k_proj_weight.data.copy_(params['k_proj_weight'])

        params['v_proj_weight'] = torch.randn(3 * 5, meta_model.vdim)
        model.v_proj_weight.data.copy_(params['v_proj_weight'])

    else:
        params['in_proj_weight'] = torch.randn(3 * 3 * 5, 3 * 5)
        model.in_proj_weight.data.copy_(params['in_proj_weight'])

    if bias:
        params['in_proj_bias'] = torch.randn(3 * 3 * 5)
        model.in_proj_bias.data.copy_(params['in_proj_bias'])

    if add_bias_kv:
        params['bias_k'] = torch.randn(1, 1, 3 * 5)
        model.bias_k.data.copy_(params['bias_k'])

        params['bias_v'] = torch.randn(1, 1, 3 * 5)
        model.bias_v.data.copy_(params['bias_v'])

    params['out_proj.weight'] = torch.randn(3 * 5, 3 * 5)
    model.out_proj.weight.data.copy_(params['out_proj.weight'])

    params['out_proj.bias'] = torch.randn(3 * 5)
    model.out_proj.bias.data.copy_(params['out_proj.bias'])

    query = torch.randn(13, 17, 3 * 5)
    key = torch.randn(19, 17, 3 * 5 if (kdim is None) else kdim)
    value = torch.randn(19, 17, 3 * 5 if (vdim is None) else vdim)

    outputs_torchmeta, weights_torchmeta = meta_model(query, key, value, params=params)
    outputs_nn, weights_nn = model(query, key, value)

    np.testing.assert_equal(outputs_torchmeta.detach().numpy(),
                            outputs_nn.detach().numpy())
    np.testing.assert_equal(weights_torchmeta.detach().numpy(),
                            weights_nn.detach().numpy())


@pytest.mark.parametrize('bias', [True, False])
@pytest.mark.parametrize('add_bias_kv', [True, False])
@pytest.mark.parametrize('kdim,vdim', [(None, None), (7, 11)])
def test_metamultiheadattention_meta_named_parameters(bias, add_bias_kv, kdim, vdim):
    meta_model = MetaMultiheadAttention(3 * 5, 3,
                                        bias=bias,
                                        add_bias_kv=add_bias_kv,
                                        kdim=kdim, vdim=vdim)
    params = OrderedDict(meta_model.meta_named_parameters())
    param_names = set(params)

    if not meta_model._qkv_same_embed_dim:
        assert {'q_proj_weight', 'k_proj_weight', 'v_proj_weight'} <= param_names
    else:
        assert 'in_proj_weight' in param_names

    if bias:
        'in_proj_bias' in param_names

    if add_bias_kv:
        assert {'bias_k', 'bias_v'} <= param_names

    assert {'out_proj.weight', 'out_proj.bias'} <= param_names
