# torchmeta
A collection of extensions and data-loaders for few-shot learning & meta-learning in [PyTorch](https://pytorch.org/). The package contains popular meta-learning benchmarks, fully compatible with both [`torchvision`](https://pytorch.org/docs/stable/torchvision/index.html) and PyTorch's [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

### Example
This minimal example below shows how to create a dataloader for the 5-shot 5-way Omniglot dataset with `torchmeta`. The dataloader loads a batch of randomly generated tasks. For more examples, check the [examples](examples/) folder.
```python
from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter
from torchvision.transforms import Resize, ToTensor, Compose

from torchmeta.utils.data import BatchMetaDataLoader

dataset = Omniglot('data', num_classes_per_task=5,
                   transform=Compose([Resize(28), ToTensor()]),
                   target_transform=Categorical(num_classes=5),
                   meta_train=True, download=True)
dataset = ClassSplitter(dataset, num_train_per_class=5, num_test_per_class=15)

dataloader = BatchMetaDataLoader(dataset, batch_size=16, num_workers=4)

for batch in dataloader:
    train_inputs, train_targets = batch['train']
    print('Train inputs shape: {0}'.format(train_inputs.shape))
    print('Train targets shape: {0}'.format(train_targets.shape))
    # Train inputs shape: torch.Size([16, 25, 1, 28, 28])
    # Train targets shape: torch.Size([16, 25])

    test_inputs, test_targets = batch['test']
    print('Test inputs shape: {0}'.format(test_inputs.shape))
    print('Test targets shape: {0}'.format(test_targets.shape))
    # Test inputs shape: torch.Size([16, 75, 1, 28, 28])
    # Test targets shape: torch.Size([16, 75])
```
