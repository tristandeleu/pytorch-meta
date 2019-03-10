import torch
from itertools import combinations
from torchvision.transforms import Compose

from torchmeta.tasks import ConcatTask

def fixed_category(index):
    def _fixed_category(i):
        return index
    return _fixed_category

class ClassDataset(object):
    def __init__(self, class_augmentations=None):
        if class_augmentations is not None:
            if not isinstance(class_augmentations, list):
                raise ValueError()
            class_augmentations = [transform for augmentations
                in class_augmentations for transform in augmentations]
        else:
            class_augmentations = []
        self.class_augmentations = class_augmentations

    def get_transform(self, index, transform):
        transform_index = (index // self.num_classes) - 1
        if transform_index < 0:
            return transform
        class_transform = self.class_augmentations[transform_index]
        if transform is None:
            return class_transform
        return Compose([class_transform, transform])

    def get_target_transform(self, index, transform):
        categorical_transform = fixed_category(index)
        if transform is None:
            return categorical_transform
        return Compose([transform, categorical_transform])

    def __getitem__(self, index):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_classes * (len(self.class_augmentations) + 1)


class MetaDataset(object):
    def __init__(self, dataset_transform=None):
        self.dataset_transform = dataset_transform

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def sample_task(self):
        index = torch.randint(len(self), size=()).item()
        return self[index]

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

    def sample_task(self):
        import random
        index = random.sample(range(self.dataset.num_classes), self.num_classes_per_task)
        return self[index]

    def __getitem__(self, index):
        if isinstance(index, int):
            raise ValueError('The index of a `CombinationMetaDataset` must be '
                'a tuple of integers, and not an integer. For example, call '
                '`dataset[({0})]` to get a task with classes from 0 to {1} '
                '(got `{2}`).'.format(', '.join([str(idx)
                for idx in range(self.num_classes_per_task)]),
                self.num_classes_per_task - 1, index))
        assert len(index) == self.num_classes_per_task
        datasets = [self.dataset[i] for i in index]
        task = ConcatTask(datasets, self.num_classes_per_task)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def __len__(self):
        num_classes, length = len(self.dataset), 1
        for i in range(1, self.num_classes_per_task + 1):
            length *= (num_classes - i + 1) / i
        return int(length)
