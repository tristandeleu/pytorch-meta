import warnings

from torchmeta.datasets import Omniglot, MiniImagenet, TieredImagenet
from torchmeta.transforms import CategoricalWrapper, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor

def omniglot(folder, shots, ways, shuffle=True, **kwargs):
    if 'num_classes_per_task' in kwargs:
        warnings.warn('', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(28), ToTensor()])
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = [Rotation([90, 180, 270])]

    dataset = Omniglot(folder, num_classes_per_task=ways, **kwargs)
    dataset = CategoricalWrapper(dataset)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=shots)

    return dataset

def miniimagenet(folder, shots, ways, shuffle=True, **kwargs):
    if 'num_classes_per_task' in kwargs:
        warnings.warn('', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(84), ToTensor()])
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = [Rotation([90, 180, 270])]

    dataset = MiniImagenet(folder, num_classes_per_task=ways, **kwargs)
    dataset = CategoricalWrapper(dataset)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=shots)

    return dataset

def tieredimagenet(folder, shots, ways, shuffle=True, **kwargs):
    if 'num_classes_per_task' in kwargs:
        warnings.warn('', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(84), ToTensor()])
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = [Rotation([90, 180, 270])]

    dataset = TieredImagenet(folder, num_classes_per_task=ways, **kwargs)
    dataset = CategoricalWrapper(dataset)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=shots)

    return dataset
