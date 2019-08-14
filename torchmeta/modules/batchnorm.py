import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn.modules.batchnorm import _BatchNorm
from torchmeta.modules.module import MetaModule

class _MetaBatchNorm(_BatchNorm, MetaModule):
    def forward(self, input, params=None):
        self._check_input_dim(input)
        if params is None:
            params = OrderedDict(self.named_parameters())

        # exponential_average_factor is self.momentum set to
        # (when it is available) only so that if gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        weight = params.get('weight', None)
        bias = params.get('bias', None)

        return F.batch_norm(
            input, self.running_mean, self.running_var, weight, bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

class MetaBatchNorm1d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm1d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))

class MetaBatchNorm2d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm2d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class MetaBatchNorm3d(_MetaBatchNorm):
    __doc__ = nn.BatchNorm3d.__doc__

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
