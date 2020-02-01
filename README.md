# Torchmeta
[![PyPI](https://img.shields.io/pypi/v/torchmeta)](https://pypi.org/project/torchmeta/) [![Build Status](https://travis-ci.com/tristandeleu/pytorch-meta.svg?branch=master)](https://travis-ci.com/tristandeleu/pytorch-meta) [![Documentation](https://img.shields.io/badge/docs-torchmeta-blue)](https://tristandeleu.github.io/pytorch-meta/)

A collection of extensions and data-loaders for few-shot learning & meta-learning in [PyTorch](https://pytorch.org/). Torchmeta contains popular meta-learning benchmarks, fully compatible with both [`torchvision`](https://pytorch.org/docs/stable/torchvision/index.html) and PyTorch's [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

#### Features
  - A unified interface for both few-shot classification and regression problems, to allow easy benchmarking on multiple problems and reproducibility.
  - Helper functions for some popular problems, with default arguments from the literature.
  - An thin extension of PyTorch's [`Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module), called `MetaModule`, that simplifies the creation of certain meta-learning models (e.g. gradient based meta-learning methods). See the [MAML example](examples/maml) for an example using `MetaModule`.

#### Datasets available
  - **Few-shot regression** (toy problems):
    - Sine waves ([Finn et al., 2017](https://arxiv.org/abs/1703.03400))
    - Harmonic functions ([Lacoste et al., 2018](https://arxiv.org/abs/1806.07528))
    - Sinusoid & lines ([Finn et al., 2018](https://arxiv.org/abs/1806.02817))
  - **Few-shot classification** (image classification):
    - Omniglot ([Lake et al., 2015](http://www.sciencemag.org/content/350/6266/1332.short)[, 2019](https://arxiv.org/abs/1902.03477))
    - Mini-ImageNet ([Vinyals et al., 2016](https://arxiv.org/abs/1606.04080), [Ravi et al., 2017](https://openreview.net/forum?id=rJY0-Kcll))
    - Tiered-ImageNet ([Ren et al., 2018](https://arxiv.org/abs/1803.00676))
    - CIFAR-FS ([Bertinetto et al., 2018](https://arxiv.org/abs/1805.08136))
    - Fewshot-CIFAR100 ([Oreshkin et al., 2018](https://arxiv.org/abs/1805.10123))
    - Caltech-UCSD Birds ([Hilliard et al., 2019](https://arxiv.org/abs/1802.04376), [Wah et al., 2019](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html))
    - Double MNIST ([Sun, 2019](https://github.com/shaohua0116/MultiDigitMNIST))

## Installation
You can install Torchmeta either using Python's package manager pip, or from source. To avoid any conflict with your existing Python setup, it is suggested to work in a virtual environment with [`virtualenv`](https://docs.python-guide.org/dev/virtualenvs/). To install `virtualenv`:
```bash
pip install --upgrade virtualenv
virtualenv venv
source venv/bin/activate
```

#### Requirements
 - Python 3.5 or above
 - PyTorch 1.3 or above
 - Torchvision 0.4 or above

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

## Example

#### Minimal example
This minimal example below shows how to create a dataloader for the 5-shot 5-way Omniglot dataset with Torchmeta. The dataloader loads a batch of randomly generated tasks, and all the samples are concatenated into a single tensor. For more examples, check the [examples](examples/) folder.
```python
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader

dataset = omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

for batch in dataloader:
    train_inputs, train_targets = batch["train"]
    print('Train inputs shape: {0}'.format(train_inputs.shape))    # (16, 25, 1, 28, 28)
    print('Train targets shape: {0}'.format(train_targets.shape))  # (16, 25)

    test_inputs, test_targets = batch["test"]
    print('Test inputs shape: {0}'.format(test_inputs.shape))      # (16, 75, 1, 28, 28)
    print('Test targets shape: {0}'.format(test_targets.shape))    # (16, 75)
```

#### Advanced example
Helper functions are only available for some of the datasets available. However, all of them are available through the unified interface provided by Torchmeta. The variable `dataset` defined above is equivalent to the following
```python
from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader

dataset = Omniglot("data",
                   # Number of ways
                   num_classes_per_task=5,
                   # Resize the images to 28x28 and converts them to PyTorch tensors (from Torchvision)
                   transform=Compose([Resize(28), ToTensor()]),
                   # Transform the labels to integers (e.g. ("Glagolitic/character01", "Sanskrit/character14", ...) to (0, 1, ...))
                   target_transform=Categorical(num_classes=5),
                   # Creates new virtual classes with rotated versions of the images (from Santoro et al., 2016)
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_train=True,
                   download=True)
dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=5, num_test_per_class=15)
dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)
```
Note that the dataloader, receiving the dataset, remains the same.

## Citation
> Tristan Deleu, Tobias Würfl, Mandana Samiei, Joseph Paul Cohen, and Yoshua Bengio. Torchmeta: A Meta-Learning library for PyTorch, 2019 [[ArXiv](https://arxiv.org/abs/1909.06576)]

If you want to cite Torchmeta, use the following Bibtex entry:
```
@misc{deleu2019torchmeta,
  title={{Torchmeta: A Meta-Learning library for PyTorch}},
  author={Deleu, Tristan and W\"urfl, Tobias and Samiei, Mandana and Cohen, Joseph Paul and Bengio, Yoshua},
  year={2019},
  url={https://arxiv.org/abs/1909.06576},
  note={Available at: https://github.com/tristandeleu/pytorch-meta}
}
```
