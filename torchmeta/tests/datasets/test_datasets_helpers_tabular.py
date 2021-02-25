import pytest

import os
import torch

from torchmeta.utils.data import MetaDataset, Task
from torchmeta.datasets import helpers_tabular

is_local = (os.getenv('TORCHMETA_DATA_FOLDER') is not None)


@pytest.mark.skipif(not is_local, reason='Requires datasets downloaded locally')
@pytest.mark.parametrize('name', helpers_tabular.__all__)
@pytest.mark.parametrize('shots', [5, 1])  # large number first for `bach' dataset.
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_datasets_helpers_tabular(name, shots, split):
    function = getattr(helpers_tabular, name)
    folder = os.getenv('TORCHMETA_DATA_FOLDER')
    download = bool(os.getenv('TORCHMETA_DOWNLOAD', False))
    test_shots = 10
    ways = 5

    dataset = function(folder,
                       ways=ways,
                       shots=shots,
                       test_shots=test_shots,
                       meta_split=split,
                       download=download)

    assert isinstance(dataset, MetaDataset)

    task = dataset.sample_task()

    # Task is a dictionary with keys [train, test]
    assert isinstance(task, dict)
    assert set(task.keys()) == set(['train', 'test'])

    # Train
    assert isinstance(task['train'], Task)
    assert task['train'].num_classes == ways
    assert len(task['train']) == ways * shots

    # Test
    assert isinstance(task['test'], Task)
    assert task['test'].num_classes == ways
    assert len(task['test']) == ways * test_shots
