from torch.nn import DataParallel as DataParallel_
from torchmeta.modules.module import MetaModule

from torch.nn.parallel import parallel_apply
from torch.nn.parallel.scatter_gather import scatter_kwargs


class DataParallel(DataParallel_, MetaModule):
    __doc__ = DataParallel_.__doc__

    def scatter(self, inputs, kwargs, device_ids):
        try:
            params = kwargs.pop('params')
        except KeyError:
            return super(DataParallel, self).scatter(inputs, kwargs, device_ids)

        inputs_, kwargs_ = scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
        # Add params argument unchanged back in kwargs
        kwargs_ = tuple(k.update(params=params) for k in kwargs_)
        return inputs_, kwargs_
