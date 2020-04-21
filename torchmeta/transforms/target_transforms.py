from torchvision.transforms import Compose, Resize, ToTensor
import PIL

class SegmentationPairTransform(object):
    def __init__(self, target_size):
        self.image_transform = Compose([Resize((target_size, target_size)), ToTensor()])
        self.mask_transform = Compose([Resize((target_size, target_size),
                                               interpolation=PIL.Image.NEAREST),
                                       ToTensor()])

    def __call__(self, image, mask):
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        return image, mask

class TargetTransform(object):
    def __call__(self, target):
        raise NotImplementedError()

    def __repr__(self):
        return str(self.__class__.__name__)


class DefaultTargetTransform(TargetTransform):
    def __init__(self, class_augmentations):
        super(DefaultTargetTransform, self).__init__()
        self.class_augmentations = class_augmentations

        self._augmentations = dict((augmentation, i + 1)
            for (i, augmentation) in enumerate(class_augmentations))
        self._augmentations[None] = 0

    def __call__(self, target):
        assert isinstance(target, tuple) and len(target) == 2
        label, augmentation = target
        return (label, self._augmentations[augmentation])
