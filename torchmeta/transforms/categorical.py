import torch
from torchmeta.tasks import TaskWrapper
from torchmeta.transforms.utils import apply_wrapper
from collections import defaultdict


class CategoricalWrapper_(TaskWrapper):
    def __init__(self, task):
        super(CategoricalWrapper_, self).__init__(task)
        self._classes = None
        self._labels = torch.randperm(self.num_classes).tolist()

    @property
    def classes(self):
        if self._classes is None:
            default_factory = lambda: self._labels[len(self._classes)]
            self._classes = defaultdict(None)
            self._classes.default_factory = default_factory
        if len(self._classes) > self.num_classes:
            raise ValueError()
        return self._classes

    def __getitem__(self, index):
        sample = self.task[index]
        if (not isinstance(sample, tuple)) or (len(sample) < 2):
            raise ValueError()
        return sample[:-1] + (self.classes[sample[-1]],)


def CategoricalWrapper(task=None):
    """Wraps a task, or a meta-dataset, so that the examples from the task(s) 
    have a target in [0, num_classes) (instead of the raw target from the 
    dataset).

    Parameters
    ----------
    task : `Task` instance or `MetaDataset` instance, optional
        If `task` is a `Task` instance, then wraps the task. If `task` is a 
        `MetaDataset` instance, then adds it to its `dataset_transform`. If 
        `None`, returns a function to be used as a `dataset_transform`.

    Examples
    --------
    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True)
    >>> task = dataset.sample_task()
    >>> task[0]
    (<PIL.Image.Image image mode=L size=105x105 at 0x11EC797F0>,
    ('images_evaluation/Glagolitic/character12', None))

    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True)
    >>> task = dataset.sample_task()
    >>> task = CategoricalWrapper(task)
    >>> task[0]
    (<PIL.Image.Image image mode=L size=105x105 at 0x11ED3F668>, 2)

    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True)
    >>> dataset = CategoricalWrapper(dataset)
    >>> task = dataset.sample_task()
    >>> task[0]
    (<PIL.Image.Image image mode=L size=105x105 at 0x11EA55E10>, 4)
    """
    class Categorical(object):
        def __call__(self, task):
            return CategoricalWrapper_(task)

        def __repr__(self):
            return '{0}()'.format(self.__class__.__name__)
    return apply_wrapper(Categorical(), task)


class FixedCategory(object):
    def __init__(self, transform=None):
        self.transform = transform

    def __call__(self, index):
        return (index, self.transform)

    def __repr__(self):
        return ('{0}({1})'.format(self.__class__.__name__, self.transform))
