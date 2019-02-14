import os
from PIL import Image

from torchvision.datasets import Omniglot as TorchvisionOmniglot
from torchmeta.dataset import Dataset

class Omniglot(Dataset, TorchvisionOmniglot):
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
