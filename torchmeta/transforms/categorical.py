import torch
from torchmeta.tasks import TaskWrapper, Task
from torchmeta.transforms.utils import apply_wrapper
from torchvision.transforms import Compose
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
