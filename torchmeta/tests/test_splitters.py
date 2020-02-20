import pytest

import numpy as np
from collections import OrderedDict

from torchmeta.transforms.splitters import ClassSplitter
from torchmeta.toy import Sinusoid
from torchmeta.utils.data import Task

def test_seed_class_splitter():
    dataset_transform = ClassSplitter(shuffle=True,
        num_train_per_class=5, num_test_per_class=5)
    dataset = Sinusoid(10, num_tasks=1000, noise_std=0.1,
        dataset_transform=dataset_transform)
    dataset.seed(1)

    expected_train_inputs = np.array([-2.03870077,  0.09898378,  3.75388738,  1.08565437, -1.56211897])
    expected_train_targets = np.array([-0.1031986 , -1.61885041,  0.91773121, -0.00309463, -1.37650356])

    expected_test_inputs = np.array([ 4.62078213, -2.48340416,  0.32922559,  0.76977846, -3.15504396])
    expected_test_targets = np.array([-0.9346262 ,  0.73113509, -1.52508997, -0.4698061 ,  1.86656819])

    task = dataset[0]
    train_dataset, test_dataset = task['train'], task['test']

    assert len(train_dataset) == 5
    assert len(test_dataset) == 5

    for i, (train_input, train_target) in enumerate(train_dataset):
        assert np.isclose(train_input, expected_train_inputs[i])
        assert np.isclose(train_target, expected_train_targets[i])

    for i, (test_input, test_target) in enumerate(test_dataset):
        assert np.isclose(test_input, expected_test_inputs[i])
        assert np.isclose(test_target, expected_test_targets[i])

def test_class_splitter_for_fold_overlaps():
    class DemoTask(Task):
        def __init__(self):
            super(DemoTask, self).__init__(index=0, num_classes=None)
            self._inputs = np.arange(10)

        def __len__(self):
            return len(self._inputs)

        def __getitem__(self, index):
            return self._inputs[index]

    splitter = ClassSplitter(shuffle=True, num_train_per_class=5, num_test_per_class=5)
    task = DemoTask()

    all_train_samples = list()
    all_test_samples = list()

    # split task ten times into train and test
    for i in range(10):
        tasks_split = splitter(task)
        train_task = tasks_split["train"]
        test_task = tasks_split["test"]

        train_samples = set([train_task[i] for i in range(len(train_task))])
        test_samples = set([test_task[i] for i in range(len(train_task))])

        # no overlap between train and test splits at single split
        assert len(train_samples.intersection(test_samples)) == 0

        all_train_samples.append(train_samples)
        all_train_samples.append(train_samples)

    # gather unique samples from multiple splits
    samples_in_all_train_splits = set().union(*all_train_samples)
    samples_in_all_test_splits = set().union(*all_test_samples)

    # no overlap between train and test splits at multiple splits
    assert len(samples_in_all_test_splits.intersection(samples_in_all_train_splits)) == 0