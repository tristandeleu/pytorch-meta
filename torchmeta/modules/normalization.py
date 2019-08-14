import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules.module import MetaModule

class MetaLayerNorm(nn.LayerNorm, MetaModule):
    __doc__ = nn.LayerNorm.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)
