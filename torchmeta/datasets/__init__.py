from torchmeta.datasets.triplemnist import TripleMNIST
from torchmeta.datasets.doublemnist import DoubleMNIST
from torchmeta.datasets.cub import CUB
from torchmeta.datasets.cifar100 import CIFARFS, FC100
from torchmeta.datasets.miniimagenet import MiniImagenet
from torchmeta.datasets.omniglot import Omniglot
from torchmeta.datasets.tieredimagenet import TieredImagenet
from torchmeta.datasets.tcga import TCGA
from torchmeta.datasets.pascal5i import Pascal5i
from torchmeta.datasets.letter_task_id_6 import Letter
from torchmeta.datasets.one_hundred_plants_texture import PlantsTexture

from torchmeta.datasets import helpers
from torchmeta.datasets import helpers_tabular

__all__ = [
    'TCGA',
    'Omniglot',
    'MiniImagenet',
    'TieredImagenet',
    'CIFARFS',
    'FC100',
    'CUB',
    'DoubleMNIST',
    'TripleMNIST',
    'Pascal5i',
    'helpers',
    'Letter',
    'PlantsTexture',
    'helpers_tabular'
]
