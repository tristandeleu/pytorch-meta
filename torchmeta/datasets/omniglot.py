import os
from PIL import Image

from torchvision.datasets import Omniglot as TorchvisionOmniglot
from torchmeta.dataset import Dataset, Task

class DeprecatedOmniglot(Dataset, TorchvisionOmniglot):
    def __init__(self, root, meta_train=True, transform=None,
                 class_transforms=None, download=False):
        TorchvisionOmniglot.__init__(self, root, background=meta_train,
                                     transform=transform, download=download)
        Dataset.__init__(self, class_transforms=class_transforms)
        self._num_classes = len(self._characters)

    @property
    def num_classes(self):
        return self._num_classes

    def get_length(self, char_index):
        return len(self._character_images[char_index % self.num_classes])

    def __getitem__(self, index):
        char_index, class_index, index = index
        images = self._character_images[char_index % self.num_classes]
        assert index < len(images)
        image_name, _ = images[index]
        image_filename = os.path.join(self.target_folder,
            self._characters[char_index % self.num_classes], image_name)
        image = Image.open(image_filename, mode='r').convert('L')

        image = self.class_transform(char_index, image)
        if self.transform:
            image = self.transform(image)

        return image, class_index

class Omniglot(Dataset, TorchvisionOmniglot):
    def __init__(self, root, meta_train=True, transform=None,
                 target_transform=None, class_transforms=None, download=False):
        TorchvisionOmniglot.__init__(self, root, background=meta_train,
                                     transform=transform, download=download)
        Dataset.__init__(self, class_transforms=class_transforms)
        self._num_classes = len(self._characters)

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        character = self._characters[index % self.num_classes]
        images = self._character_images[index % self.num_classes]
        class_transform = self.class_transform(index)

        return OmniglotTask(self.target_folder, character, images,
            transform=self.transform, target_transform=self.target_transform,
            class_transform=class_transform)

class OmniglotTask(Task):
    def __init__(self, folder, character, images, transform=None,
                 target_transform=None, class_transform=None):
        super(OmniglotTask, self).__init__()
        self.folder = folder
        self.character = character
        self.images = images

        self.transform = transform
        self.target_transform = target_transform
        self.class_transform = class_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name, target = self.images[index]
        filename = os.path.join(self.folder, self.character, name)
        image = Image.open(filename, mode='r').convert('L')

        if self.class_transform is not None:
            image = self.class_transform(image)
        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)

if __name__ == '__main__':
    dataset = Omniglot('data')
    task = dataset[1]

    print(task.images)
    print(dataset._character_images[1])
    # dataset._character_images[1] = None
    task.images = None
    print(dataset._character_images[1])
    print(task.images)
