from collections import OrderedDict

from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset

from torchmeta.utils.data.dataset import CombinationMetaDataset
from torchmeta.utils.data.sampler import (CombinationSequentialSampler,
                                          CombinationRandomSampler)

def batch_meta_collate(collate_fn):
    def collate_task(task):
        if isinstance(task, TorchDataset):
            return collate_fn([task[idx] for idx in range(len(task))])
        elif isinstance(task, OrderedDict):
            return OrderedDict([(key, collate_task(subtask))
                for (key, subtask) in task.items()])
        else:
            raise NotImplementedError()

    def _collate_fn(batch):
        return collate_fn([collate_task(task) for task in batch])

    return _collate_fn

def no_collate(batch):
    return batch

class MetaDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None):
        if collate_fn is None:
            collate_fn = no_collate

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


class BatchMetaDataLoader(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        collate_fn = batch_meta_collate(default_collate)

        super(BatchMetaDataLoader, self).__init__(dataset,
            batch_size=batch_size, shuffle=shuffle, sampler=sampler,
            batch_sampler=None, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)
