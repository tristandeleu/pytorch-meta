import torch

from collections import OrderedDict, defaultdict
from torchmeta.tasks import TaskWrapper, ConcatTask, SubsetTask


class Splitter(object):
    def __init__(self, splits):
        self.splits = splits

    def get_indices(self, task):
        raise NotImplementedError()

    def __call__(self, task):
        indices = self.get_indices(task)
        return OrderedDict([(split, SubsetTask(task, indices[split]))
            for split in self.splits])

    def __len__(self):
        return len(self.splits)


class ClassSplitter(Splitter):
    def __init__(self, shuffle=False, num_samples_per_class=None,
                 num_train_per_class=None, num_test_per_class=None):
        self.shuffle = shuffle
        if num_samples_per_class is None:
            num_samples_per_class = OrderedDict()
            if num_train_per_class is not None:
                num_samples_per_class['train'] = num_train_per_class
            if num_test_per_class is not None:
                num_samples_per_class['test'] = num_test_per_class
        assert len(num_samples_per_class) > 0
        self._min_samples_per_class = min(num_samples_per_class.values())
        super(ClassSplitter, self).__init__(num_samples_per_class)

    def get_indices_task(self, task):
        all_class_indices = defaultdict(list)
        for index in range(len(task)):
            sample = task[index]
            if (not isinstance(sample, tuple)) or (len(sample) < 2):
                raise ValueError()
            all_class_indices[sample[-1]].append(index)
        if len(all_class_indices) != task.num_classes:
            raise ValueError()

        indices = OrderedDict([(split, []) for split in self.splits])
        for class_indices in all_class_indices.values():
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class:
                raise ValueError()
            if self.shuffle:
                dataset_indices = torch.randperm(num_samples).tolist()
            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = (dataset_indices[ptr:ptr + num_split]
                    if self.shuffle else range(ptr, ptr + num_split))
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split
        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0
        for dataset in task.unwrapped.datasets:
            num_samples = len(dataset)
            if num_samples < self._min_samples_per_class:
                raise ValueError()
            if self.shuffle:
                dataset_indices = torch.randperm(num_samples).tolist()
            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = (dataset_indices[ptr:ptr + num_split]
                    if self.shuffle else range(ptr, ptr + num_split))
                indices[split].extend([idx + cum_size for idx in split_indices])
                ptr += num_split
            cum_size += num_samples
        return indices

    def get_indices(self, task):
        if isinstance(task.unwrapped, ConcatTask):
            if len(task) != len(task.unwrapped):
                # TODO: raise a warning that we might have already taken a
                # subset of the original task. Roll back to get_indices_task
                return self.get_indices_task(task)
            indices = self.get_indices_concattask(task)
        elif isinstance(task, Task):
            indices = self.get_indices_task(task)
        else:
            raise ValueError()
        return indices
