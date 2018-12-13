import torch
from torch.utils.data.sampler import Sampler, BatchSampler

from torchmeta.dataset import Dataset

class DatasetSampler(Sampler):
    def __init__(self, data_source, class_sampler, num_classes,
                 train_size_per_class, test_size_per_class=None, shuffle=False):
        if not isinstance(data_source, Dataset):
            raise ValueError()
        self.data_source = data_source
        self.class_sampler = class_sampler
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.train_size_per_class = train_size_per_class
        if test_size_per_class is None:
            test_size_per_class = 0
        self.test_size_per_class = test_size_per_class

        self._train_dataset_size = self.num_classes * self.train_size_per_class
        self._test_dataset_size = self.num_classes * self.test_size_per_class

    def __iter__(self):
        train_dataset, test_dataset = [], []
        for way_index, class_index in enumerate(self.class_sampler):
            length = self.data_source.get_length(class_index)
            assert self.train_size_per_class + self.test_size_per_class < length
            indices = torch.randperm(length).tolist()

            class_indices = (class_index, way_index % self.num_classes)
            train_dataset.extend([class_indices + (index,)
                for index in indices[:self.train_size_per_class]])
            test_dataset.extend([class_indices + (index,)
                for index in indices[self.train_size_per_class:
                self.train_size_per_class + self.test_size_per_class]])

            if len(train_dataset) == self._train_dataset_size:
                if self.shuffle:
                    train_dataset = [train_dataset[index] for index in
                        torch.randperm(self._train_dataset_size).tolist()]
                    if self.test_size_per_class > 0:
                        test_dataset = [test_dataset[index] for index in
                            torch.randperm(self._test_dataset_size).tolist()]
                yield (train_dataset, test_dataset)
                train_dataset, test_dataset = [], []

class BatchDatasetSampler(BatchSampler):
    def __iter__(self):
        train_indices, test_indices = [], []
        batch_length = 0
        for train_dataset, test_dataset in self.sampler:
            train_indices.extend(train_dataset)
            test_indices.extend(test_dataset)
            batch_length += 1
            if batch_length == self.batch_size:
                yield train_indices + test_indices
                train_indices, test_indices = [], []
                batch_length = 0
        if batch_length > 0 and not self.drop_last:
            yield train_indices + test_indices
