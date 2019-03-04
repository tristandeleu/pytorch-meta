import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision.datasets import Omniglot as TorchvisionOmniglot
from torchmeta.dataset import ClassDataset, CombinationMetaDataset

class Omniglot(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=True,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_transforms=None, download=False):
        dataset = OmniglotClassDataset(root, meta_train=meta_train,
            transform=transform, target_transform=target_transform,
            class_transforms=class_transforms, download=download)
        super(Omniglot, self).__init__(dataset, num_classes_per_task,
            dataset_transform=dataset_transform)


class OmniglotClassDataset(ClassDataset, TorchvisionOmniglot):
    def __init__(self, root, meta_train=True, transform=None,
                 target_transform=None, class_transforms=None, download=False):
        TorchvisionOmniglot.__init__(self, root, background=meta_train,
            transform=transform, target_transform=None, download=download)
        ClassDataset.__init__(self, class_transforms=class_transforms)
        self._num_classes = len(self._characters)

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        character = self._characters[index % self.num_classes]
        images = self._character_images[index % self.num_classes]
        transform = self.get_class_transform(index, self.transform)
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
