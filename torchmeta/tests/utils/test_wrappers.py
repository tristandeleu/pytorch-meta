import pytest

import os
import torch
from torch.utils.data import Dataset, DataLoader

from torchmeta.datasets import helpers
from torchmeta.utils.data import NonEpisodicWrapper

is_local = (os.getenv('TORCHMETA_DATA_FOLDER') is not None)


@pytest.mark.skipif(not is_local, reason='Requires datasets downloaded locally')
@pytest.mark.parametrize('name', helpers.__all__)
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
def test_datasets_helpers_wrapper(name, split):
    function = getattr(helpers, name)
    folder = os.getenv('TORCHMETA_DATA_FOLDER')
    download = bool(os.getenv('TORCHMETA_DOWNLOAD', False))

    dataset = function(folder,
                       ways=5,
                       shots=1,
                       test_shots=15,
                       meta_split=split,
                       download=download)

    wrapped_dataset = NonEpisodicWrapper(dataset)

    assert isinstance(wrapped_dataset, Dataset)

    img, label = wrapped_dataset[0]

    dataloader = DataLoader(wrapped_dataset, batch_size=4)
    inputs, targets = next(iter(dataloader))

    assert isinstance(inputs, torch.Tensor)
    assert inputs.ndim == 4
    assert inputs.shape[0] == 4

    assert len(targets) == 2
    assert isinstance(targets[1], torch.Tensor)
