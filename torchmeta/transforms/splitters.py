import torch
import numpy as np

from collections import OrderedDict, defaultdict
from torchmeta.utils.data.task import Task, ConcatTask, SubsetTask
from torchmeta.transforms.utils import apply_wrapper

__all__ = ['Splitter', 'ClassSplitter', 'WeightedClassSplitter']


class Splitter(object):
    def __init__(self, splits, random_state_seed):
        self.splits = splits
        self.random_state_seed = random_state_seed
        self.seed(random_state_seed)

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed=seed)

    def get_indices(self, task):
        if isinstance(task, ConcatTask):
            indices = self.get_indices_concattask(task)
        elif isinstance(task, Task):
            indices = self.get_indices_task(task)
        else:
            raise ValueError('The task must be of type `ConcatTask` or `Task`, '
                'Got type `{0}`.'.format(type(task)))
        return indices

    def get_indices_task(self, task):
        raise NotImplementedError('Method `get_indices_task` must be '
            'implemented in classes inherited from `Splitter`.')

    def get_indices_concattask(self, task):
        raise NotImplementedError('Method `get_indices_concattask` must be '
            'implemented in classes inherited from `Splitter`.')

    def _get_class_indices(self, task):
        class_indices = defaultdict(list)
        if task.num_classes is None: # Regression task
            class_indices['regression'] = range(len(task))
        else:
            for index in range(len(task)):
                sample = task[index]
                if (not isinstance(sample, tuple)) or (len(sample) < 2):
                    raise ValueError('In order to split the dataset in train/'
                        'test splits, `Splitter` must access the targets. Each '
                        'sample from a task must be a tuple with at least 2 '
                        'elements, with the last one being the target.')
                class_indices[sample[-1]].append(index)

            if len(class_indices) != task.num_classes:
                raise ValueError('The number of classes detected in `Splitter` '
                    '({0}) is different from the property `num_classes` ({1}) '
                    'in task `{2}`.'.format(len(class_indices),
                    task.num_classes, task))

        return class_indices

    def __call__(self, task):
        indices = self.get_indices(task)
        return OrderedDict([(split, SubsetTask(task, indices[split]))
            for split in self.splits])

    def __len__(self):
        return len(self.splits)


class ClassSplitter_(Splitter):
    def __init__(self, shuffle=True, num_samples_per_class=None,
                 num_train_per_class=None, num_test_per_class=None,
                 num_support_per_class=None, num_query_per_class=None,
                 random_state_seed=0):
        """
        Transforms a dataset into train/test splits for few-shot learning tasks,
        based on a fixed number of samples per class for each split. This is a
        dataset transformation to be applied as a `dataset_transform` in a
        `MetaDataset`.

        Parameters
        ----------
        shuffle : bool (default: `True`)
            Shuffle the data in the dataset before the split.

        num_samples_per_class : dict, optional
            Dictionary containing the names of the splits (as keys) and the
            corresponding number of samples per class in each split (as values).
            If not `None`, then the arguments `num_train_per_class`,
            `num_test_per_class`, `num_support_per_class` and
            `num_query_per_class` are ignored.

        num_train_per_class : int, optional
            Number of samples per class in the training split. This corresponds
            to the number of "shots" in "k-shot learning". If not `None`, this
            creates an item `train` for each task.

        num_test_per_class : int, optional
            Number of samples per class in the test split. If not `None`, this
            creates an item `test` for each task.

        num_support_per_class : int, optional
            Alias for `num_train_per_class`. If `num_train_per_class` is not
            `None`, then this argument is ignored. If not `None`, this creates
            an item `support` for each task.

        num_query_per_class : int, optional
            Alias for `num_test_per_class`. If `num_test_per_class` is not
            `None`, then this argument is ignored. If not `None`, this creates
            an item `query` for each task.

        random_state_seed : int, optional
            seed of the np.RandomState. Defaults to '0'.

        Examples
        --------
        >>> transform = ClassSplitter(num_samples_per_class={
        ...     'train': 5, 'test': 15})
        >>> dataset = Omniglot('data', num_classes_per_task=5,
        ...                    dataset_transform=transform, meta_train=True)
        >>> task = dataset.sample_task()
        >>> task.keys()
        ['train', 'test']
        >>> len(task['train']), len(task['test'])
        (25, 75)
        """
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
        super(ClassSplitter_, self).__init__(num_samples_per_class, random_state_seed)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        for name, class_indices in all_class_indices.items():
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for class `{0}` ({1}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({2}).'.format(name,
                    num_samples, self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split

        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        for dataset in task.datasets:
            num_samples = len(dataset)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for one class ({0}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({1}).'.format(num_samples,
                    self._min_samples_per_class))

            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, num_split in self.splits.items():
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend(split_indices + cum_size)
                ptr += num_split
            cum_size += num_samples

        return indices


class WeightedClassSplitter_(Splitter):
    def __init__(self, shuffle=True, min_num_samples=1, max_num_samples=None,
                 weights=None, train_weights=None, test_weights=None,
                 support_weights=None, query_weights=None,
                 force_equal_per_class=False, random_state_seed=0):
        """
        Transforms a dataset into train/test splits for few-shot learning tasks.
        The number of samples per class is proportional to the number of samples
        per class in the original dataset. This is a dataset transformation to
        be applied as a `dataset_transform` in a `MetaDataset`.

        Parameters
        ----------
        shuffle : bool (default: `True`)
            Shuffle the data in the dataset before the split.

        min_num_samples : int or dict, optional (default: 1)
            Minimum number of samples per class.

        max_num_samples : int or dict, optional
            Maximum number of samples per class.

        weights : dict, optional
            Dictionary containing the names of the splits (as keys) and the
            corresponding proportions of samples per class in each split (as
            values). If not `None`, then the arguments `train_weights`,
            `test_weights`, `support_weights` and `query_weights` are ignored.

        train_weights : float, optional
            Proportion of samples from each class in the training split. If not
            `None`, this creates an item `train` for each task.

        test_weights : float, optional
            Proportion of samples from each class in the training split. If not
            `None`, this creates an item `test` for each task.

        support_weights : float, optional
            Alias for `train_weights`. If `train_weights` is not `None`, then
            this argument is ignored. If not `None`, this creates an item
            `support` for each task.

        query_weights : float, optional
            Alias for `test_weights`. If `test_weights` is not `None`, then this
            argument is ignored. If not `None`, this creates an item `query` for
            each task.

        force_equal_per_class : bool (default: `False`)
            If `True`, then the number of samples per class is equal for each
            class; this is then proportional to the number of samples in the
            class with the minimum number of samples.

        random_state_seed : int, optional
            seed of the np.RandomState. Defaults to '0'.
        """
        self.shuffle = shuffle
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
            raise NotImplementedError('Argument `min_num_samples` in '
                '`WeightedClassSplitter` must be of type `dict` or `int`. Got '
                'type `{0}`.'.format(type(min_num_samples)))

        if max_num_samples is None:
            self.max_num_samples = None
        elif isinstance(max_num_samples, int):
            self.max_num_samples = OrderedDict([(split, max_num_samples)
                for split in weights])
        elif isinstance(max_num_samples, dict):
            self.max_num_samples = OrderedDict(max_num_samples)
        else:
            raise NotImplementedError('Argument `max_num_samples` in '
                '`WeightedClassSplitter` must be of type `dict` or `int`. Got '
                'type `{0}`.'.format(type(min_num_samples)))

        self._min_samples_per_class = sum(self.min_num_samples.values())
        super(WeightedClassSplitter_, self).__init__(weights, random_state_seed)

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        min_samples = min([len(class_indices) for class_indices
            in all_class_indices.values()])
        if min_samples < self._min_samples_per_class:
            raise ValueError('The smallest number of samples in a class ({0}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `WeightedClassSplitter` ({1}).'.format(
                    min_samples, self._min_samples_per_class))

        for class_indices in all_class_indices.values():
            num_samples = (min_samples if self.force_equal_per_class
                else len(class_indices))
            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, weight in self.splits.items():
                num_split = max(self.min_num_samples[split], int(weight * num_samples))
                if self.max_num_samples is not None:
                    num_split = min(self.max_num_samples[split], num_split)
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split

        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        min_samples = min([len(dataset) for dataset in task.datasets])
        if min_samples < self._min_samples_per_class:
            raise ValueError('The smallest number of samples in a class ({0}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `WeightedClassSplitter` ({1}).'.format(
                    min_samples, self._min_samples_per_class))

        for dataset in task.datasets:
            num_samples = (min_samples if self.force_equal_per_class
                else len(dataset))
            if self.shuffle:
                seed = (hash(task) + self.random_state_seed) % (2 ** 32)
                dataset_indices = np.random.RandomState(seed).permutation(num_samples)
            else:
                dataset_indices = np.arange(num_samples)

            ptr = 0
            for split, weight in self.splits.items():
                num_split = max(self.min_num_samples, int(weight * num_samples))
                split_indices = dataset_indices[ptr:ptr + num_split]
                if self.shuffle:
                    self.np_random.shuffle(split_indices)
                indices[split].extend(split_indices + cum_size)
            cum_size += num_samples

        return indices


def ClassSplitter(task=None, *args, **kwargs):
    return apply_wrapper(ClassSplitter_(*args, **kwargs), task)

def WeightedClassSplitter(task=None, *args, **kwargs):
    return apply_wrapper(WeightedClassSplitter_(*args, **kwargs), task)
