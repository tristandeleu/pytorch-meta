import torch

from collections import OrderedDict, defaultdict
from torchmeta.tasks import TaskWrapper, Task, ConcatTask, SubsetTask
from torchmeta.transforms.utils import apply_wrapper


class Splitter(object):
    def __init__(self, splits):
        self.splits = splits

    def get_indices(self, task):
        if isinstance(task.unwrapped, ConcatTask):
            if len(task) != len(task.unwrapped):
                import warnings
                warnings.warn('The length of the transformed task is different '
                    'from the length of the original task. Maybe one of the '
                    'dataset transformations applied is already taking a subset '
                    'of the task (eg. calling two splitters in '
                    '`dataset_transform`). `ClassSplitter` will roll back to a '
                    'simple stategy for dataset splitting.', UserWarning, stacklevel=2)
                return self.get_indices_task(task)
            indices = self.get_indices_concattask(task)
        elif isinstance(task, Task):
            indices = self.get_indices_task(task)
        else:
            raise ValueError()
        return indices

    def get_indices_task(self, task):
        raise NotImplementedError()

    def get_indices_concattask(self, task):
        raise NotImplementedError()

    def _get_class_indices(self, task):
        class_indices = defaultdict(list)
        if task.num_classes is None: # Regression task
            class_indices['regression'] = range(len(task))
        else:
            for index in range(len(task)):
                sample = task[index]
                if (not isinstance(sample, tuple)) or (len(sample) < 2):
                    raise ValueError()
                class_indices[sample[-1]].append(index)
            if len(class_indices) != task.num_classes:
                raise ValueError()
        return class_indices

    def __call__(self, task):
        indices = self.get_indices(task)
        return OrderedDict([(split, SubsetTask(task, indices[split]))
            for split in self.splits])

    def __len__(self):
        return len(self.splits)


class ClassSplitter_(Splitter):
    def __init__(self, shuffle=False, num_samples_per_class=None,
                 num_train_per_class=None, num_test_per_class=None,
                 num_support_per_class=None, num_query_per_class=None):
        self.shuffle = shuffle
        if num_samples_per_class is None:
            num_samples_per_class = OrderedDict()
            if num_train_per_class is not None:
                num_samples_per_class['train'] = num_train_per_class
            elif num_support_per_class is not None:
                num_samples_per_class['support'] = num_support_per_class
            if num_test_per_class is not None:
                num_samples_per_class['test'] = num_test_per_class
            elif num_query_per_class is not None:
                num_samples_per_class['query'] = num_query_per_class
        assert len(num_samples_per_class) > 0
        self._min_samples_per_class = sum(num_samples_per_class.values())
        super(ClassSplitter_, self).__init__(num_samples_per_class)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
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


class WeightedClassSplitter_(Splitter):
    def __init__(self, shuffle=True, min_num_samples=1, max_num_samples=None,
                 weights=None, train_weights=None, test_weights=None,
                 support_weights=None, query_weights=None,
                 force_equal_per_class=False):
        self.shuffle = shuffle
        self.min_num_samples = min_num_samples
        self.force_equal_per_class = force_equal_per_class

        if weights is None:
            weights = OrderedDict()
            if train_weights is not None:
                weights['train'] = train_weights
            elif support_weights is not None:
                weights['support'] = support_weights
            if test_weights is not None:
                weights['test'] = test_weights
            elif query_weights is not None:
                weights['query'] = query_weights
        assert len(weights) > 0
        assert sum(weights.values()) <= 1.

        if (min_num_samples is None) or isinstance(min_num_samples, int):
            if min_num_samples is None:
                min_num_samples = 0
            self.min_num_samples = OrderedDict([(split, min_num_samples)
                for split in weights])
        elif isinstance(min_num_samples, dict):
            self.min_num_samples = OrderedDict(min_num_samples)
        else:
            raise NotImplementedError()

        if max_num_samples is None:
            self.max_num_samples = None
        elif isinstance(max_num_samples, int):
            self.max_num_samples = OrderedDict([(split, max_num_samples)
                for split in weights])
        elif isinstance(max_num_samples, dict):
            self.max_num_samples = OrderedDict(max_num_samples)
        else:
            raise NotImplementedError()

        self._min_samples_per_class = sum(self.min_num_samples.values())
        super(WeightedClassSplitter_, self).__init__(weights)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])
        min_samples = min([len(class_indices) for class_indices
            in all_class_indices.values()])
        if min_samples < self._min_samples_per_class:
            raise ValueError()
        for class_indices in all_class_indices.values():
            num_samples = (min_samples if self.force_equal_per_class
                else len(class_indices))
            if self.shuffle:
                dataset_indices = torch.randperm(num_samples).tolist()
            ptr = 0
            for split, weight in self.splits.items():
                num_split = max(self.min_num_samples[split], int(weight * num_samples))
                if self.max_num_samples is not None:
                    num_split = min(self.max_num_samples[split], num_split)
                split_indices = (dataset_indices[ptr:ptr + num_split]
                    if self.shuffle else range(ptr, ptr + num_split))
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split
        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0
        min_samples = min([len(dataset) for dataset in task.unwrapped.datasets])
        if min_samples < self._min_samples_per_class:
            raise ValueError()
        for dataset in task.unwrapped.datasets:
            num_samples = (min_samples if self.force_equal_per_class
                else len(dataset))
            if num_samples < self._min_samples_per_class:
                raise ValueError()
            if self.shuffle:
                dataset_indices = torch.randperm(num_samples).tolist()
            ptr = 0
            for split, weight in self.splits.items():
                num_split = max(self.min_num_samples, int(weight * num_samples))
                split_indices = (dataset_indices[ptr:ptr + num_split]
                    if self.shuffle else range(ptr, ptr + num_split))
                indices[split].extend([idx + cum_size for idx in split_indices])
            cum_size += num_samples
        return indices


def ClassSplitter(task=None, *args, **kwargs):
    return apply_wrapper(ClassSplitter_(*args, **kwargs), task)

def WeightedClassSplitter(task=None, *args, **kwargs):
    return apply_wrapper(WeightedClassSplitter_(*args, **kwargs), task)
