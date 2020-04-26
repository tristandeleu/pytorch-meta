import pytest

import numpy as np
from collections import OrderedDict

from torchmeta.utils.data import Task, MetaDataset
from torchmeta.toy import Sinusoid, Harmonic, SinusoidAndLine
from torchmeta.toy import helpers


@pytest.mark.parametrize('dataset_class',
    [Sinusoid, Harmonic, SinusoidAndLine])
def test_toy_meta_dataset(dataset_class):
    dataset = dataset_class(10, num_tasks=1000, noise_std=None)

    assert isinstance(dataset, MetaDataset)
    assert len(dataset) == 1000


@pytest.mark.parametrize('dataset_class',
    [Sinusoid, Harmonic, SinusoidAndLine])
def test_toy_task(dataset_class):
    dataset = dataset_class(10, num_tasks=1000, noise_std=None)
    task = dataset[0]

    assert isinstance(task, Task)
    assert len(task) == 10


@pytest.mark.parametrize('dataset_class',
    [Sinusoid, Harmonic, SinusoidAndLine])
def test_toy_sample(dataset_class):
    dataset = dataset_class(10, num_tasks=1000, noise_std=None)
    task = dataset[0]
    input, target = task[0]

    assert isinstance(input, np.ndarray)
    assert isinstance(target, np.ndarray)
    assert input.shape == (1,)
    assert target.shape == (1,)


@pytest.mark.parametrize('name,dataset_class',
    [('sinusoid', Sinusoid), ('harmonic', Harmonic)])
def test_toy_helpers(name, dataset_class):
    dataset_fn = getattr(helpers, name)
    dataset = dataset_fn(shots=5, test_shots=15)
    assert isinstance(dataset, dataset_class)

    task = dataset[0]
    assert isinstance(task, OrderedDict)
    assert 'train' in task
    assert 'test' in task

    train, test = task['train'], task['test']
    assert isinstance(train, Task)
    assert isinstance(test, Task)
    assert len(train) == 5
    assert len(test) == 15
