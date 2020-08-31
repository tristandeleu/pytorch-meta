import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules.module import MetaModule
from torchmeta.modules.linear import MetaLinear

class MetaMultiheadAttention(nn.MultiheadAttention, MetaModule):
    __doc__ = nn.MultiheadAttention.__doc__

    def __init__(self, *args, **kwargs):
        super(MetaMultiheadAttention, self).__init__(*args, **kwargs)
        self.out_proj = MetaLinear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        in_proj_weight = params.get('in_proj_weight', None)
        in_proj_bias = params.get('in_proj_bias', None)
        bias_k = params.get('bias_k', None)
        bias_v = params.get('bias_v', None)

        if not self._qkv_same_embed_dim:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, params['out_proj.weight'], params['out_proj.bias'],
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=params['q_proj_weight'],
                k_proj_weight=params['k_proj_weight'],
                v_proj_weight=params['v_proj_weight'])
        else:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, in_proj_bias,
                bias_k, bias_v, self.add_zero_attn,
                self.dropout, params['out_proj.weight'], params['out_proj.bias'],
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
