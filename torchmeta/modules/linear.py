import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch._jit_internal import weak_module, weak_script_method
from torchmeta.modules.module import MetaModule

@weak_module
class MetaLinear(nn.Linear, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias)

@weak_module
class MetaBilinear(nn.Bilinear, MetaModule):
    @weak_script_method
    def forward(self, input1, input2, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.bilinear(input1, input2, params['weight'], bias)
