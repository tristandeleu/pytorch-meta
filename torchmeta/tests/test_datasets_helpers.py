import pytest

import os
import torch

from torchmeta.utils.data import MetaDataset, Task
from torchmeta.datasets import helpers

is_local = (os.getenv('TORCHMETA_DATA_FOLDER') is not None)


@pytest.mark.skipif(not is_local, reason='Requires datasets downloaded locally')
@pytest.mark.parametrize('name', helpers.__all__)
@pytest.mark.parametrize('shots', [1, 5])
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_datasets_helpers(name, shots, split):
    function = getattr(helpers, name)
    folder = os.getenv('TORCHMETA_DATA_FOLDER')
    download = bool(os.getenv('TORCHMETA_DOWNLOAD', False))

    dataset = function(folder,
                       ways=5,
                       shots=shots,
                       test_shots=15,
                       meta_split=split,
                       download=download)

    assert isinstance(dataset, MetaDataset)

    task = dataset.sample_task()

    # Task is a dictionary with keys [train, test]
    assert isinstance(task, dict)
    assert set(task.keys()) == set(['train', 'test'])

    # Train
    assert isinstance(task['train'], Task)
    assert task['train'].num_classes == 5
    assert len(task['train']) == 5 * shots

    # Test
    assert isinstance(task['test'], Task)
    assert task['test'].num_classes == 5
    assert len(task['test']) == 5 * 15 # test_shots
