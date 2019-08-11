import random
from itertools import combinations
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torchmeta.utils.data.dataset import CombinationMetaDataset

__all__ = ['CombinationSequentialSampler', 'CombinationRandomSampler']


class CombinationSequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise ValueError()
        super(CombinationSequentialSampler, self).__init__(data_source)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        return combinations(range(num_classes), num_classes_per_task)


class CombinationRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise ValueError()
        self.data_source = data_source

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            yield tuple(random.sample(range(num_classes), num_classes_per_task))
