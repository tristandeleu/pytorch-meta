import warnings

from torchmeta.datasets import Letter, PlantsTexture, PlantsShape, PlantsMargin, Bach
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.transforms.tabular_transforms import NumpyToTorch

__all__ = [
    'letter',
    'plants_texture',
    'plants_shape',
    'plants_margin',
    'bach'
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


def letter(folder: str, shots: int, ways: int, shuffle: bool=True,
              test_shots: int=None, seed: int=None, **kwargs) -> Letter:
    """
    Wrapper that creates a meta-dataset for the Letter tabular dataset.

    Notes
    --------
    Letter has 26 classes in total with default splits train/val/test : 15/5/6.

    See also
    --------
    `datasets.Letter` : CombinationMetaDataset for the Letter dataset.
    """
    return helper_with_default_tabular(Letter, folder, shots, ways, shuffle=shuffle,
                                       test_shots=test_shots, seed=seed, defaults=None, **kwargs)


def plants_texture(folder: str, shots: int, ways: int, shuffle: bool=True,
                   test_shots: int=None, seed: int=None, **kwargs) -> PlantsTexture:
    """
    Wrapper that creates a meta-dataset for the PlantsTexture tabular dataset.

    Notes
    --------
    PlantsTexture has 100 classes in total with default splits train/val/test : 70/15/15.

    See also
    --------
    `datasets.PlantsTexture` : CombinationMetaDataset for the PlantsTexture dataset.
    """
    return helper_with_default_tabular(PlantsTexture, folder, shots, ways, shuffle=shuffle,
                                       test_shots=test_shots, seed=seed, defaults=None, **kwargs)


def plants_shape(folder: str, shots: int, ways: int, shuffle: bool=True,
                 test_shots: int=None, seed: int=None, **kwargs) -> PlantsShape:
    """
    Wrapper that creates a meta-dataset for the PlantsShape tabular dataset.

    Notes
    --------
    PlantsShape has 100 classes in total with default splits train/val/test : 70/15/15.

    See also
    --------
    `datasets.PlantsShape` : Meta-dataset for the PlantsShape dataset.
    """
    return helper_with_default_tabular(PlantsShape, folder, shots, ways, shuffle=shuffle,
                                       test_shots=test_shots, seed=seed, defaults=None, **kwargs)


def plants_margin(folder: str, shots: int, ways: int, shuffle: bool=True,
                  test_shots: int=None, seed: int=None, **kwargs) -> PlantsMargin:
    """
    Wrapper that creates a meta-dataset for the PlantsMargin tabular dataset.

    Notes
    --------
    PlantsMargin has 100 classes in total with default splits train/val/test : 70/15/15.

    See also
    --------
    `datasets.PlantsMargin` : CombinationMetaDataset for the PlantsMargin dataset.
    """
    return helper_with_default_tabular(PlantsMargin, folder, shots, ways, shuffle=shuffle,
                                       test_shots=test_shots, seed=seed, defaults=None, **kwargs)


def bach(folder: str, shots: int, ways: int, shuffle: bool=True, test_shots: int=None,
         min_num_samples_per_class: int=None, seed: int=None, **kwargs) -> Bach:
    """
    Wrapper that creates a meta-dataset for the Bach tabular dataset.

    Notes
    --------
    Bach has 101 classes in total with default splits train/val/test : 70/15/15. # Todo change

    See also
    --------
    `datasets.Bach` : CombinationMetaDataset for the Bach dataset.
    """
    if min_num_samples_per_class is None:
        if test_shots is None:
            min_num_samples_per_class = int(2 * shots)
        else:
            min_num_samples_per_class = int(test_shots + shots)
    return helper_with_default_tabular(Bach, folder, shots, ways, shuffle=shuffle,
                                       test_shots=test_shots, seed=seed, defaults=None,
                                       min_num_samples_per_class=min_num_samples_per_class, **kwargs)
