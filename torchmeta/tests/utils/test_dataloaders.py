import pytest

import os
import torch
from torch.utils.data import DataLoader

from torchmeta.datasets import helpers
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import Task, MetaDataLoader, BatchMetaDataLoader

is_local = (os.getenv('TORCHMETA_DATA_FOLDER') is not None)


def test_meta_dataloader():
    dataset = Sinusoid(10, num_tasks=1000, noise_std=None)
    meta_dataloader = MetaDataLoader(dataset, batch_size=4)
    assert isinstance(meta_dataloader, DataLoader)
    assert len(meta_dataloader) == 250 # 1000 / 4
    
    batch = next(iter(meta_dataloader))
    assert isinstance(batch, list)
    assert len(batch) == 4

    task = batch[0]
    assert isinstance(task, Task)
    assert len(task) == 10


def test_meta_dataloader_task_loader():
    dataset = Sinusoid(10, num_tasks=1000, noise_std=None)
    meta_dataloader = MetaDataLoader(dataset, batch_size=4)
    batch = next(iter(meta_dataloader))

    dataloader = DataLoader(batch[0], batch_size=5)
    inputs, targets = next(iter(dataloader))

    assert len(dataloader) == 2 # 10 / 5
    # PyTorch dataloaders convert numpy array to tensors
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.shape == (5, 1)
    assert targets.shape == (5, 1)


def test_batch_meta_dataloader():
    dataset = Sinusoid(10, num_tasks=1000, noise_std=None)
    meta_dataloader = BatchMetaDataLoader(dataset, batch_size=4)
    assert isinstance(meta_dataloader, DataLoader)
    assert len(meta_dataloader) == 250 # 1000 / 4

    inputs, targets = next(iter(meta_dataloader))
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    assert inputs.shape == (4, 10, 1)
    assert targets.shape == (4, 10, 1)


def test_batch_meta_dataloader_splitter():
    dataset = Sinusoid(20, num_tasks=1000, noise_std=None)
    dataset = ClassSplitter(dataset, num_train_per_class=5,
        num_test_per_class=15)
    meta_dataloader = BatchMetaDataLoader(dataset, batch_size=4)

    batch = next(iter(meta_dataloader))
    assert isinstance(batch, dict)
    assert 'train' in batch
    assert 'test' in batch

    train_inputs, train_targets = batch['train']
    test_inputs, test_targets = batch['test']

    assert isinstance(train_inputs, torch.Tensor)
    assert isinstance(train_targets, torch.Tensor)
    assert train_inputs.shape == (4, 5, 1)
    assert train_targets.shape == (4, 5, 1)

    assert isinstance(test_inputs, torch.Tensor)
    assert isinstance(test_targets, torch.Tensor)
    assert test_inputs.shape == (4, 15, 1)
    assert test_targets.shape == (4, 15, 1)


@pytest.mark.skipif(not is_local, reason='Requires datasets downloaded locally')
@pytest.mark.parametrize('name', helpers.__all__)
@pytest.mark.parametrize('shots', [1, 5])
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_datasets_helpers_dataloader(name, shots, split):
    function = getattr(helpers, name)
    folder = os.getenv('TORCHMETA_DATA_FOLDER')
    download = bool(os.getenv('TORCHMETA_DOWNLOAD', False))

    dataset = function(folder,
                       ways=5,
                       shots=shots,
                       test_shots=15,
                       meta_split=split,
                       download=download)

    meta_dataloader = BatchMetaDataLoader(dataset, batch_size=4)

    batch = next(iter(meta_dataloader))
    assert isinstance(batch, dict)
    assert 'train' in batch
    assert 'test' in batch

    train_inputs, train_targets = batch['train']
    test_inputs, test_targets = batch['test']

    assert isinstance(train_inputs, torch.Tensor)
    assert isinstance(train_targets, torch.Tensor)
    assert train_inputs.ndim == 5
    assert train_inputs.shape[:2] == (4, 5 * shots)
    assert train_targets.ndim == 2
    assert train_targets.shape[:2] == (4, 5 * shots)

    assert isinstance(test_inputs, torch.Tensor)
    assert isinstance(test_targets, torch.Tensor)
    assert test_inputs.ndim == 5
    assert test_inputs.shape[:2] == (4, 5 * 15) # test_shots
    assert test_targets.ndim == 2
    assert test_targets.shape[:2] == (4, 5 * 15)


@pytest.mark.skipif(not is_local, reason='Requires datasets downloaded locally')
def test_overflow_length_dataloader():
    folder = os.getenv('TORCHMETA_DATA_FOLDER')
    download = bool(os.getenv('TORCHMETA_DOWNLOAD', False))

    # The number of tasks is C(4112, 20), which exceeds machine precision
    dataset = helpers.omniglot(folder,
                               ways=20,
                               shots=1,
                               test_shots=5,
                               meta_train=True,
                               download=download)

    meta_dataloader = BatchMetaDataLoader(dataset, batch_size=4)

    batch = next(iter(meta_dataloader))
    assert isinstance(batch, dict)
    assert 'train' in batch
    assert 'test' in batch
