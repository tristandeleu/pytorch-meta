from torch.utils.data import Dataset, ConcatDataset, Subset

class Task(Dataset):
    """Base class for a classification task.

    Parameters
    ----------
    num_classes : int
        Number of classes for the classification task.
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @property
    def unwrapped(self):
        return self


class TaskWrapper(Task):
    """Base class for a wrapper around a classification task.

    Parameters
    ----------
    task : `Task` instance
        Task to be wrapped over.
    """
    def __init__(self, task):
        assert isinstance(task, Task)
        super(TaskWrapper, self).__init__(task.num_classes)
        self.task = task

    def __len__(self):
        return len(self.task)

    @property
    def unwrapped(self):
        return self.task.unwrapped


class ConcatTask(Task, ConcatDataset):
    def __init__(self, datasets, num_classes):
        Task.__init__(self, num_classes)
        ConcatDataset.__init__(self, datasets)

    def __getitem__(self, index):
        return ConcatDataset.__getitem__(self, index)


class SubsetTask(Task, Subset):
    def __init__(self, dataset, indices, num_classes=None):
        if num_classes is None:
            num_classes = dataset.num_classes
        Task.__init__(self, num_classes)
        Subset.__init__(self, dataset, indices)

    def __getitem__(self, index):
        return Subset.__getitem__(self, index)
