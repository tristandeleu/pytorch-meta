from itertools import combinations

class Dataset(object):
    def __init__(self, class_transforms=None):
        if class_transforms is not None:
            if not isinstance(class_transforms, list):
                raise ValueError()
            class_transforms = [transform for class_transform
                in class_transforms for transform in class_transform]
        else:
            class_transforms = []
        self.class_transforms = class_transforms

    def class_transform(self, index):
        transform_index = (index // self.num_classes) - 1
        if transform_index < 0:
            return None
        return self.class_transforms[transform_index]

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_classes * (len(self.class_transforms) + 1)


class TaskDataset(object):
    def __init__(self, meta_dataset, classes_per_task):
        super(TaskDataset, self).__init__()
        self.meta_dataset = meta_dataset
        self.classes_per_task = classes_per_task

    def __iter__(self):
        num_classes = len(self.meta_dataset)
        for index in combinations(num_classes, self.classes_per_task):
            yield self[index]

    def __getitem__(self, index):
        assert len(index) == self.classes_per_task
        datasets = tuple(self.meta_dataset[i] for i in index)
        return ConcatTask(datasets)


class Task(object):
    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
