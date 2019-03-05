import torch
from itertools import combinations
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose

class ClassDataset(object):
    def __init__(self, class_transforms=None):
        if class_transforms is not None:
            if not isinstance(class_transforms, list):
                raise ValueError()
            class_transforms = [transform for class_transform
                in class_transforms for transform in class_transform]
        else:
            class_transforms = []
        self.class_transforms = class_transforms

    def get_class_transform(self, index, transform):
        transform_index = (index // self.num_classes) - 1
        if transform_index < 0:
            return transform
        class_transform = self.class_transforms[transform_index]
        if transform is None:
            return class_transform
        return Compose([class_transform, transform])

    def get_target_transform(self, index, transform):
        categorical_transform = lambda _: index
        if transform is None:
            return categorical_transform
        return Compose([transform, categorical_transform])

    def __getitem__(self, index):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_classes * (len(self.class_transforms) + 1)


class MetaDataset(object):
    def __init__(self, dataset_transform=None):
        self.dataset_transform = dataset_transform

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class CombinationMetaDataset(MetaDataset):
    def __init__(self, dataset, num_classes_per_task, dataset_transform=None):
        super(CombinationMetaDataset, self).__init__(dataset_transform=dataset_transform)
        if not isinstance(num_classes_per_task, int):
            raise ValueError()
        self.dataset = dataset
        self.num_classes_per_task = num_classes_per_task

    def __iter__(self):
        num_classes = len(self.dataset)
        for index in combinations(num_classes, self.num_classes_per_task):
            yield self[index]

    def __getitem__(self, index):
        assert len(index) == self.num_classes_per_task
        datasets = [self.dataset[i] for i in index]
        task = ConcatTask(datasets, self.num_classes_per_task)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def __len__(self):
        from scipy.special import binom
        num_classes = len(self.dataset)
        return int(binom(num_classes, self.num_classes_per_task))


class Task(Dataset):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    @property
    def unwrapped(self):
        return self


class TaskWrapper(Task):
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
