import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules.module import MetaModule

class MetaLinear(nn.Linear, MetaModule):
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.linear(input, params['weight'], bias)

class MetaBilinear(nn.Bilinear, MetaModule):
    def forward(self, input1, input2, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)
        return F.bilinear(input1, input2, params['weight'], bias)
