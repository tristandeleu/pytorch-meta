import numpy as np
import io

from PIL import Image
from torch.utils.data import Dataset

from torchmeta.utils.data.dataset import CombinationMetaDataset


class NonEpisodicWrapper(Dataset):
    """Non-episodic wrapper to convert a CombinationMetaDataset into a standard
    PyTorch Dataset, compatible with (non-episodic) training.

    Parameters
    ----------
    dataset : `CombinationMetaDataset` instance
        The meta-dataset to be wrapped around.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.
    """
    def __init__(self, dataset, target_transform=None):
        super(NonEpisodicWrapper, self).__init__()
        if not isinstance(dataset, CombinationMetaDataset):
            raise ValueError('`NonEpisodicWrapper` can only be wrapped around a '
                '`CombinationMetaDataset`. The dataset `{0}` is not an instance '
                'of `CombinationMetaDataset`.'.format(dataset))

        self.dataset = dataset
        self.target_transform = target_transform

        class_dataset = self.dataset.dataset
        self._labels, pointer = [], 0
        self._offsets = np.zeros((class_dataset.num_classes,), dtype=np.int_)
        for index, label in enumerate(class_dataset.labels):
            if isinstance(label, list):
                label = '/'.join(label)
            num_samples = len(class_dataset.data[label])
            self._labels.append(label)
            self._offsets[index] = pointer
            pointer += num_samples
        self._num_samples = pointer

    def __getitem__(self, index):
        class_dataset = self.dataset.dataset

        class_index = np.maximum(np.searchsorted(self._offsets,
            index % self.num_samples, side='left') - 1, 0)
        offset = (index % self.num_samples) - self._offsets[class_index]
        label = self._labels[class_index]

        array = class_dataset.data[label][offset]
        image = (Image.open(io.BytesIO(array))
            if array.ndim < 2 else Image.fromarray(array))

        class_augmented_index = (class_dataset.num_classes
            * (index // self.num_samples) + class_index)
        transform = class_dataset.get_transform(class_augmented_index,
                                                class_dataset.transform)
        if transform is not None:
            image = transform(image)

        class_transform = class_dataset.get_class_augmentation(class_augmented_index)
        label = (label, index // self.num_samples)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_classes(self):
        num_augmentations = len(self.dataset.dataset.class_augmentations)
        return len(self._labels) * (num_augmentations + 1)

    def __len__(self):
        num_augmentations = len(self.dataset.dataset.class_augmentations)
        return self.num_samples * (num_augmentations + 1)
