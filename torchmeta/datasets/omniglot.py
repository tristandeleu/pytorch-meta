import os
import json
import glob
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import Omniglot as TorchvisionOmniglot
from torchmeta.dataset import ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import list_dir
from torchmeta.datasets.utils import get_asset

class Omniglot(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, use_vinyals_split=True,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_transforms=None, meta_split=None,
                 class_augmentations=None, download=False):
        if class_transforms is not None:
            import warnings
            warnings.warn('The argument `class_transforms` is deprecated. '
                'Please use the argument `class_augmentations` instead.',
                DeprecationWarning, stacklevel=2)
            if class_augmentations is None:
                class_augmentations = class_transforms
        dataset = OmniglotClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            use_vinyals_split=use_vinyals_split, transform=transform,
            target_transform=target_transform, meta_split=meta_split,
            class_augmentations=class_augmentations, download=download)
        super(Omniglot, self).__init__(dataset, num_classes_per_task,
            dataset_transform=dataset_transform)


class OmniglotClassDataset(ClassDataset, TorchvisionOmniglot):
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 use_vinyals_split=True, transform=None, target_transform=None,
                 class_augmentations=None, meta_split=None, download=False):
        ClassDataset.__init__(self, meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        if self.meta_val and (not use_vinyals_split):
            raise ValueError()

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform
        self.target_transform = target_transform
        self.background = (self.meta_train or self.meta_val)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.use_vinyals_split:
            data = get_asset('omniglot', '{0}.json'.format(self.meta_split), dtype='json')
            self._characters = [os.path.join(
                'images_{0}'.format(folder), alphabet, character)
                for (folder, alphabets) in data.items()
                for (alphabet, characters) in alphabets.items()
                for character in characters]

            if self.meta_train:
                # The training split contains alphabets from both images_background
                # and images_evaluations. We need to ensure that both datasets
                # are downloaded.
                self.background = False
                if download:
                    self.download()

                if not self._check_integrity():
                    raise RuntimeError('Dataset not found or corrupted.' +
                                       ' You can use download=True to download it')
                self.background = True
        else:
            self.background = self.meta_train
            target_folder = self._get_target_folder()
            alphabets = list_dir(os.path.join(self.root, target_folder))
            self._characters = [os.path.join(target_folder, alphabet, character)
                for alphabet in alphabets
                for character in list_dir(os.path.join(self.root, target_folder, alphabet))]

        self._character_images = [[(image, idx) for image in glob.glob(
            os.path.join(self.root, character, '*.png'))]
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

        return OmniglotDataset(character, images, transform=transform,
            target_transform=target_transform)


class OmniglotDataset(Dataset):
    def __init__(self, character, images, transform=None, target_transform=None):
        super(OmniglotDataset, self).__init__()
        self.character = character
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename, target = self.images[index]
        image = Image.open(filename, mode='r').convert('L')

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
