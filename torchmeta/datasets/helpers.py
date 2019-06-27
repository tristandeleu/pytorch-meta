import warnings

from torchmeta.datasets import Omniglot, MiniImagenet, TieredImagenet
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor

def omniglot(folder, shots, ways, shuffle=True, test_shots=None, **kwargs):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds 
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way` 
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the 
        number of test examples is equal to the number of training examples per 
        class.

    kwargs
        Additional arguments passed to the `Omniglot` class.

    See also
    --------
    `datasets.Omniglot` : Meta-dataset for the Omniglot dataset.
    """
    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
            'set in the helper function for the number of classes per task. '
            'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(28), ToTensor()])
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = Categorical(ways)
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = [Rotation([90, 180, 270])]
    if test_shots is None:
        test_shots = shots

    dataset = Omniglot(folder, num_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=test_shots)

    return dataset

def miniimagenet(folder, shots, ways, shuffle=True, test_shots=None, **kwargs):
    """Helper function to create a meta-dataset for the Mini-Imagenet dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `miniimagenet` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds 
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way` 
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the 
        number of test examples is equal to the number of training examples per 
        class.

    kwargs
        Additional arguments passed to the `MiniImagenet` class.

    See also
    --------
    `datasets.MiniImagenet` : Meta-dataset for the Mini-Imagenet dataset.
    """
    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
            'set in the helper function for the number of classes per task. '
            'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(84), ToTensor()])
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = Categorical(ways)
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = [Rotation([90, 180, 270])]
    if test_shots is None:
        test_shots = shots

    dataset = MiniImagenet(folder, num_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=test_shots)

    return dataset

def tieredimagenet(folder, shots, ways, shuffle=True, test_shots=None, **kwargs):
    """Helper function to create a meta-dataset for the Tiered-Imagenet dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `tieredimagenet` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds 
        to `k` in `k-shot` classification.

    ways : int
        Number of classes per task. This corresponds to `N` in `N-way` 
        classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the 
        number of test examples is equal to the number of training examples per 
        class.

    kwargs
        Additional arguments passed to the `TieredImagenet` class.

    See also
    --------
    `datasets.TieredImagenet` : Meta-dataset for the Tiered-Imagenet dataset.
    """
    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
            'set in the helper function for the number of classes per task. '
            'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(84), ToTensor()])
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = Categorical(ways)
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = [Rotation([90, 180, 270])]
    if test_shots is None:
        test_shots = shots

    dataset = TieredImagenet(folder, num_classes_per_task=ways, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=test_shots)

    return dataset
