import random
import warnings
from itertools import combinations
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torchmeta.utils.data.dataset import CombinationMetaDataset, OneVsAllMetaDataset, SequenceMetaDataset

__all__ = ['CombinationSequentialSampler',
           'CombinationRandomSampler',
           'OneClassSequentialSampler',
           'OneClassRandomSampler'
           ]


class CombinationSequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        super(CombinationSequentialSampler, self).__init__(data_source)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        return combinations(range(num_classes), num_classes_per_task)


class CombinationRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, CombinationMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        # Temporarily disable the warning if the length of the length of the 
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(CombinationRandomSampler, self).__init__(data_source,
                                                           replacement=True)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        num_classes_per_task = self.data_source.num_classes_per_task
        for _ in combinations(range(num_classes), num_classes_per_task):
            yield tuple(random.sample(range(num_classes), num_classes_per_task))


class OneClassRandomSampler(RandomSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, OneVsAllMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`CombinationMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        # Temporarily disable the warning if the length of the length of the
        # dataset exceeds the machine precision. This avoids getting this
        # warning shown with MetaDataLoader, even though MetaDataLoader itself
        # does not use the length of the dataset.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            super(OneClassRandomSampler, self).__init__(data_source,
                                                        replacement=True)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        for _ in range(num_classes):
            yield random.randint(0, num_classes-1)


class OneClassSequentialSampler(SequentialSampler):
    def __init__(self, data_source):
        if not isinstance(data_source, OneVsAllMetaDataset):
            raise TypeError('Expected `data_source` to be an instance of '
                            '`OneVsAllMetaDataset`, but found '
                            '{0}'.format(type(data_source)))
        super(OneClassSequentialSampler, self).__init__(data_source)

    def __iter__(self):
        num_classes = len(self.data_source.dataset)
        for i in range(num_classes):
            yield i
