from torch.utils.data import ConcatDataset, Subset
from torch.utils.data import Dataset as Dataset_
from torchvision.transforms import Compose

__all__ = ['Dataset', 'Task', 'ConcatTask', 'SubsetTask']


class Dataset(Dataset_):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def target_transform_append(self, transform):
        if transform is None:
            return
        if self.target_transform is None:
            self.target_transform = transform
        else:
            self.target_transform = Compose([self.target_transform, transform])


class Task(Dataset):
    """Base class for a classification task.

    Parameters
    ----------
    num_classes : int
        Number of classes for the classification task.
    """
    def __init__(self, num_classes, transform=None, target_transform=None):
        super(Task, self).__init__(transform=transform,
            target_transform=target_transform)
        self.num_classes = num_classes


class ConcatTask(Task, ConcatDataset):
    def __init__(self, datasets, num_classes, target_transform=None):
        Task.__init__(self, num_classes)
        ConcatDataset.__init__(self, datasets)
        for task in self.datasets:
            task.target_transform_append(target_transform)

    def __getitem__(self, index):
        return ConcatDataset.__getitem__(self, index)


class SubsetTask(Task, Subset):
    def __init__(self, dataset, indices, num_classes=None,
                 target_transform=None):
        if num_classes is None:
            num_classes = dataset.num_classes
        Task.__init__(self, num_classes)
        Subset.__init__(self, dataset, indices)
        self.dataset.target_transform_append(target_transform)

    def __getitem__(self, index):
        return Subset.__getitem__(self, index)
