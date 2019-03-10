import os
import json
import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import Omniglot as TorchvisionOmniglot
from torchmeta.dataset import ClassDataset, CombinationMetaDataset

class Omniglot(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, use_vinyals_split=True,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_transforms=None, class_augmentations=None, download=False):
        if class_transforms is not None:
            import warnings
            warnings.warn('The argument `class_transforms` is deprecated. '
                'Please use the argument `class_augmentations` instead.',
                DeprecationWarning, stacklevel=2)
            if class_augmentations is None:
                class_augmentations = class_transforms
        dataset = OmniglotClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            use_vinyals_split=use_vinyals_split,
            transform=transform, target_transform=target_transform,
            class_augmentations=class_augmentations, download=download)
        super(Omniglot, self).__init__(dataset, num_classes_per_task,
            dataset_transform=dataset_transform)


class OmniglotClassDataset(ClassDataset, TorchvisionOmniglot):
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 use_vinyals_split=True, transform=None, target_transform=None,
                 class_augmentations=None, download=False):
        if (meta_train + meta_val + meta_test) != 1:
            raise ValueError()
        if meta_val and (not use_vinyals_split):
            raise ValueError()
        TorchvisionOmniglot.__init__(self, root, background=meta_train,
            transform=transform, target_transform=None, download=download)
        ClassDataset.__init__(self, class_augmentations=class_augmentations)
        if use_vinyals_split:
            asset_path = os.path.join(os.path.dirname(__file__), 'assets', 'omniglot')
            split = 'train' if meta_train else ('val' if meta_val else 'test')
            with open(os.path.join(asset_path, '{0}.json'.format(split)), 'r') as f:
                split_dict = json.load(f)
                self._characters = ['{0}/{1}'.format(alphabet, character)
                    for (alphabet, characters) in split_dict.item()
                    for character in characters]
                self._character_images = [[(image, idx) for image in glob.glob(
                    os.path.join(self.target_folder, character, '*.png'))]
                    for (idx, character) in enumerate(self._characters)]
        self._num_classes = len(self._characters)

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        character = self._characters[index % self.num_classes]
        images = self._character_images[index % self.num_classes]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index, self.target_transform)

        return OmniglotDataset(self.target_folder, character, images,
            transform=transform, target_transform=target_transform)


class OmniglotDataset(Dataset):
    def __init__(self, folder, character, images, transform=None,
                 target_transform=None):
        super(OmniglotDataset, self).__init__()
        self.folder = folder
        self.character = character
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name, target = self.images[index]
        filename = os.path.join(self.folder, self.character, name)
        image = Image.open(filename, mode='r').convert('L')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
