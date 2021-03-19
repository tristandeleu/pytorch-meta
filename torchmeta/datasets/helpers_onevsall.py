from torchmeta.datasets import OmniglotOneVsAll
from torchmeta.transforms import ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor

__all__ = [
    'omniglot_onevsall'
]


def helper_with_default_onevsall(klass, folder, shots, shuffle=True,
                                 test_shots=None, seed=None, defaults=None, **kwargs):
    if defaults is None:
        defaults = {}
    if 'transform' not in kwargs:
        kwargs['transform'] = defaults.get('transform', ToTensor())
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = defaults.get('class_augmentations', None)
    if test_shots is None:
        test_shots = shots
    dataset = klass(folder, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle, num_train_per_class=shots,
                            num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset

def omniglot_onevsall(folder, shots, shuffle=True, test_shots=None,
                      seed=None, **kwargs):
    """Helper function to create a meta-dataset for the OmniglotOneVsAll dataset.

    Parameters
    ----------
    folder : string
        Root directory where the dataset folder `omniglot` exists.

    shots : int
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples per class in each task. If `None`, then the
        number of test examples is equal to the number of training examples per
        class.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `OmniglotOneVsAll` class.

    See also
    --------
    `datasets.OmniglotOneVsAll` : Meta-dataset for the OmniglotOneVsAll dataset.
    """
    defaults = {
        'transform': Compose([Resize(28), ToTensor()]),
        'class_augmentations': [Rotation([90, 180, 270])]
    }

    return helper_with_default_onevsall(OmniglotOneVsAll, folder, shots,
                                        shuffle=shuffle, test_shots=test_shots,
                                        seed=seed, defaults=defaults, **kwargs)
