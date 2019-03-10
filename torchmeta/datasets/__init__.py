from torchmeta.datasets.omniglot import Omniglot
from torchmeta.datasets.tcga import TCGA
from torchmeta.datasets.tcga import TCGATask as TCGATask_

class TCGATask(TCGATask_):
    def __init__(self, *args, **kwargs):
        import warnings
        warnings.warn('Access to `TCGATask` with `torchmeta.datasets.TCGATask` '
            'is deprecated. To use `TCGATask`, please use `torchmeta.datasets.'
            'tcga.TCGATask`.', DeprecationWarning, stacklevel=3)
        super(TCGATask, self).__init__(*args, **kwargs)

__all__ = ['TCGA', 'TCGATask', 'Omniglot']
