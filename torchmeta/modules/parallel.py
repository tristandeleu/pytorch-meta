import torch
from torch.nn import DataParallel as DataParallel_
from torchmeta.modules.module import MetaModule
from collections import OrderedDict

from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs
from torch.nn.parallel.replicate import _broadcast_coalesced_reshape


class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        try:
            params = kwargs.pop('params')
        except KeyError:
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids)

        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        # Add params argument unchanged back in kwargs
        replicas = self._replicate_params(params, inputs_, device_ids,
                                          detach=not torch.is_grad_enabled())
        kwargs_ = tuple(dict(params=replica, **kwarg)
                        for (kwarg, replica) in zip(kwargs_, replicas))
        return inputs_, kwargs_

    def _replicate_params(self, params, inputs, device_ids, detach=False):
        if params is None:
            return tuple(None for _ in inputs)

        replicas = _broadcast_coalesced_reshape(list(params.values()),
                                                device_ids[:len(inputs)],
                                                detach)
        replicas = tuple(OrderedDict(zip(params.keys(), replica))
                         for replica in replicas)
        return replicas
