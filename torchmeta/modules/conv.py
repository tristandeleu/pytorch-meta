import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch._jit_internal import weak_module, weak_script_method
from torch.nn.modules.utils import _single, _pair, _triple
from torchmeta.modules.module import MetaModule

@weak_module
class MetaConv1d(nn.Conv1d, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv1d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _single(0), self.dilation, self.groups)

        return F.conv1d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)

@weak_module
class MetaConv2d(nn.Conv2d, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)

@weak_module
class MetaConv3d(nn.Conv3d, MetaModule):
    @weak_script_method
    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[2] + 1) // 2, self.padding[2] // 2,
                                (self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv3d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _triple(0), self.dilation, self.groups)

        return F.conv3d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)
