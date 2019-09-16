# Torchmeta
[![PyPI version](https://badge.fury.io/py/torchmeta.svg)](https://badge.fury.io/py/torchmeta)

A collection of extensions and data-loaders for few-shot learning & meta-learning in [PyTorch](https://pytorch.org/). Torchmeta contains popular meta-learning benchmarks, fully compatible with both [`torchvision`](https://pytorch.org/docs/stable/torchvision/index.html) and PyTorch's [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

### Installation
You can install Torchmeta either using Python's package manager pip, or from source. To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/). To install `virtualenv`:
```bash
pip install --upgrade virtualenv
virtualenv venv
source venv/bin/activate
```

#### Using pip
This is the recommended way to install Torchmeta:
```bash
pip install torchmeta
```

#### From source
You can also install Torchmeta from source. This is recommended if you want to contribute to Torchmeta.
```bash
git clone https://github.com/tristandeleu/pytorch-meta.git
cd pytorch-meta
python setup.py install
```

### Example
This minimal example below shows how to create a dataloader for the 5-shot 5-way Omniglot dataset with Torchmeta. The dataloader loads a batch of randomly generated tasks, and all the samples are concatenated into a single tensor. For more examples, check the [examples](examples/) folder.
```python
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

dataset = omniglot('data', ways=5, shots=5, test_shots=15,
                   meta_train=True, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

for batch in dataloader:
    train_inputs, train_targets = batch["train"]
    print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
    print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

    test_inputs, test_targets = batch["test"]
    print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
    print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
```
