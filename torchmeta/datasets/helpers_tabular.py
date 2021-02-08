import warnings

from torchmeta.datasets import Covertype
from torchmeta.transforms import Categorical, ClassSplitter, Co
from torchmeta.transforms.tabular_transforms import NumpyToTorch

__all__ = [
    'covertype'
]


def helper_with_default_tabular(klass, folder, shots, ways, shuffle=True,
                                test_shots=None, seed=None, defaults=None, **kwargs):
    """
    Parameters
    ----------
    klass : CombinationMetaDataset
        the class corresponding to the meta-dataset, e.g., Covertype

    folder : string
        Root directory where the dataset folder exists, e.g., `covertype_task_id_2118`.

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

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `TieredImagenet` class.

    Returns
    -------
    klass
        The meta-dataset with ClassSplitter applied, e.g., Covertype.
    """

    if defaults is None:
        defaults = {}

    if 'num_classes_per_task' in kwargs:
        warnings.warn('Both arguments `ways` and `num_classes_per_task` were '
            'set in the helper function for the number of classes per task. '
            'Ignoring the argument `ways`.', stacklevel=2)
        ways = kwargs['num_classes_per_task']

    if 'transform' not in kwargs:
        kwargs['transform'] = defaults.get('transform', NumpyToTorch())

    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = defaults.get('target_transform',
                                                  Categorical(ways))
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = defaults.get('class_augmentations', None)

    if test_shots is None:
        test_shots = shots
    dataset = klass(folder,
                    num_classes_per_task=ways,
                    **kwargs)
    dataset = ClassSplitter(dataset,
                            shuffle=shuffle,
                            num_train_per_class=shots,
                            num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset


def covertype(folder: str, shots: int, ways: int, shuffle: bool=True,
              test_shots: int=None, seed: int=None, **kwargs) -> Covertype:
    """
    Wrapper that creates a meta-dataset for the Covertype dataset.

    Covertype has 7 classes in total (3, 2, 2 split).

    See also
    --------
    `datasets.Covertype` : Meta-dataset for the Covertype dataset.
    """
    return helper_with_default_tabular(Covertype, folder, shots, ways, shuffle=shuffle,
                                       test_shots=test_shots, seed=seed, defaults=None, **kwargs)
