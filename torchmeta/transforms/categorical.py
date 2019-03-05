import torch
from torchmeta.dataset import TaskWrapper
from collections import defaultdict


class CategoricalWrapper(TaskWrapper):
    def __init__(self, task):
        super(CategoricalWrapper, self).__init__(task)
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


class CategoricalTaskTarget(object):
    def __call__(self, task):
        return CategoricalWrapper(task)

    def __repr__(self):
        return '{0}()'.format(self.__class__.__name__)
