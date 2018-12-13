import torch
from torch._six import container_abcs
from collections import namedtuple

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from torchmeta.sampler import DatasetSampler, BatchDatasetSampler

Dataset = namedtuple('Dataset', 'train test')

def meta_collate_fn(collate_fn, meta_batch_size, num_classes,
                    train_size_per_class, test_size_per_class=None):
    train_dataset_size = meta_batch_size * num_classes * train_size_per_class

    def _reshape(tensor):
        if isinstance(tensor, torch.Tensor):
            shape = tensor.shape
            return tensor.view((meta_batch_size, -1) + shape[1:])
        elif isinstance(tensor, container_abcs.Sequence):
            return [_reshape(subtensor) for subtensor in tensor]
        else:
            raise TypeError()

    def _collate_fn(batch):
        train_batch = batch[:train_dataset_size]
        test_batch = batch[train_dataset_size:]

        train_dataset = _reshape(collate_fn(train_batch))
        if test_size_per_class is not None:
            test_dataset = _reshape(collate_fn(test_batch))
        else:
            test_dataset = None

        return Dataset(train=train_dataset, test=test_dataset)

    return _collate_fn

class DataLoader(TorchDataLoader):
    def __init__(self, dataset, meta_batch_size=1, num_classes=1,
                 train_size_per_class=1, test_size_per_class=None,
                 shuffle=False, shuffle_datasets=False, num_workers=0,
                 collate_fn=default_collate, pin_memory=False,
                 drop_last=False, timeout=0):
        if shuffle:
            class_sampler = RandomSampler(dataset)
        else:
            class_sampler = SequentialSampler(dataset)

        dataset_sampler = DatasetSampler(dataset, class_sampler, num_classes,
            train_size_per_class, test_size_per_class, shuffle=shuffle_datasets)
        batch_sampler = BatchDatasetSampler(dataset_sampler,
            batch_size=meta_batch_size, drop_last=drop_last)
        
        _collate_fn = meta_collate_fn(collate_fn, meta_batch_size, num_classes,
            train_size_per_class, test_size_per_class)

        super(DataLoader, self).__init__(dataset, batch_size=1, shuffle=False,
            sampler=None, batch_sampler=batch_sampler, num_workers=num_workers,
            collate_fn=_collate_fn, pin_memory=pin_memory, drop_last=False,
            timeout=timeout, worker_init_fn=None)
