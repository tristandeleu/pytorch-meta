import torch
from torch._six import container_abcs
from collections import namedtuple
import random

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.data.dataset import Subset
from torch.utils.data.dataset import Dataset as TorchDataset

from torchmeta.sampler import DatasetSampler, BatchDatasetSampler

from torchmeta.dataset import CombinationMetaDataset, MetaDataset, Task, ConcatTask
from torchmeta.samplers import CombinationSequentialSampler, CombinationRandomSampler

Dataset = namedtuple('Dataset', 'train test')

def basic_meta_collate(collate_fn, shuffle_datasets, train_size_per_class,
                       test_size_per_class=None):
    num_samples = train_size_per_class
    if test_size_per_class is not None:
        num_samples += test_size_per_class

    def get_subset_task(task):
        assert isinstance(task, TorchDataset), type(task)
        if isinstance(task, ConcatTask):
            subsets = [get_subset_task(subtask) for subtask in task.datasets]
            if test_size_per_class is None:
                return ConcatTask(subsets, task.num_classes,
                    categorical_task_target=task.categorical_task_target)
            else:
                train_subsets, test_subsets = zip(*subsets)
            return (ConcatTask(train_subsets, task.num_classes,
                categorical_task_target=task.categorical_task_target),
                ConcatTask(test_subsets, task.num_classes,
                categorical_task_target=task.categorical_task_target))
        else:
            if shuffle_datasets:
                indices = random.sample(len(task), num_samples)
            else:
                indices = range(min(len(task), num_samples))
            if test_size_per_class is None:
                return Subset(task, indices)
            else:
                return (Subset(task, indices[:train_size_per_class]),
                    Subset(task, indices[train_size_per_class:]))

    def collate_task(task):
        assert isinstance(task, TorchDataset)
        if isinstance(task, tuple):
            train_samples = collate_task(task[0])
            test_samples = collate_task(task[1])
            return Dataset(train=train_samples, test=test_samples)
        elif isinstance(task, TorchDataset):
            return [task[index] for index in range(len(task))]
        else:
            raise NotImplementedError()

    def _collate_fn(batch):
        if not isinstance(batch[0], Task):
            raise ValueError()
        subset_batch = [get_subset_task(task) for task in batch]
        samples = [collate_fn(collate_task(task)) for task in subset_batch]
        return collate_fn(samples)

    return _collate_fn


class MetaDataLoader(TorchDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        if collate_fn is None:
            collate_fn = lambda batch: batch

        if isinstance(dataset, CombinationMetaDataset) and (sampler is None):
            if shuffle:
                sampler = CombinationRandomSampler(dataset)
            else:
                sampler = CombinationSequentialSampler(dataset)
            shuffle = False

        super(MetaDataLoader, self).__init__(dataset, batch_size=batch_size,
            shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=collate_fn,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn)


class BasicMetaDataLoader(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 train_size_per_class=1, test_size_per_class=None,
                 shuffle_datasets=False, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):
        collate_fn = basic_meta_collate(default_collate, shuffle_datasets,
            train_size_per_class, test_size_per_class=test_size_per_class)

        super(BasicMetaDataLoader, self).__init__(dataset,
            batch_size=batch_size, shuffle=shuffle, sampler=None,
            batch_sampler=None, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)
