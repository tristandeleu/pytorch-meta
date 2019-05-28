import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch._jit_internal import weak_module, weak_script_method
from torchmeta.modules.module import MetaModule

@weak_module
class MetaLayerNorm(nn.LayerNorm, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weight = params.get('weight', None)
        bias = params.get('bias', None)
        return F.layer_norm(
            input, self.normalized_shape, weight, bias, self.eps)
