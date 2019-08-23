import warnings

from torchmeta.toy import Sinusoid, Harmonic
from torchmeta.transforms import ClassSplitter

def sinusoid(shots, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Sinusoid toy dataset.

    Parameters
    ----------
    shots : int
        Number of (training) examples in each task. This corresponds to `k` in
        `k-shot` classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples in each task. If `None`, then the number of test
        examples is equal to the number of training examples in each task.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Sinusoid` class.

    See also
    --------
    `torchmeta.toy.Sinusoid` : Meta-dataset for the Sinusoid toy dataset.
    """
    if 'num_samples_per_task' in kwargs:
        warnings.warn('Both arguments `shots` and `num_samples_per_task` were '
            'set in the helper function for the number of samples in each task. '
            'Ignoring the argument `shots`.', stacklevel=2)
        if test_shots is not None:
            shots = kwargs['num_samples_per_task'] - test_shots
            if shots <= 0:
                raise ValueError('The argument `test_shots` ({0}) is greater '
                    'than the number of samples per task ({1}). Either use the '
                    'argument `shots` instead of `num_samples_per_task`, or '
                    'increase the value of `num_samples_per_task`.'.format(
                    test_shots, kwargs['num_samples_per_task']))
        else:
            shots = kwargs['num_samples_per_task'] // 2
    if test_shots is None:
        test_shots = shots

    dataset = Sinusoid(num_samples_per_task=shots + test_shots, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset

def harmonic(shots, shuffle=True, test_shots=None, seed=None, **kwargs):
    """Helper function to create a meta-dataset for the Harmonic toy dataset.

    Parameters
    ----------
    shots : int
        Number of (training) examples in each task. This corresponds to `k` in
        `k-shot` classification.

    shuffle : bool (default: `True`)
        Shuffle the examples when creating the tasks.

    test_shots : int, optional
        Number of test examples in each task. If `None`, then the number of test
        examples is equal to the number of training examples in each task.

    seed : int, optional
        Random seed to be used in the meta-dataset.

    kwargs
        Additional arguments passed to the `Harmonic` class.

    See also
    --------
    `torchmeta.toy.Harmonic` : Meta-dataset for the Harmonic toy dataset.
    """
    if 'num_samples_per_task' in kwargs:
        warnings.warn('Both arguments `shots` and `num_samples_per_task` were '
            'set in the helper function for the number of samples in each task. '
            'Ignoring the argument `shots`.', stacklevel=2)
        if test_shots is not None:
            shots = kwargs['num_samples_per_task'] - test_shots
            if shots <= 0:
                raise ValueError('The argument `test_shots` ({0}) is greater '
                    'than the number of samples per task ({1}). Either use the '
                    'argument `shots` instead of `num_samples_per_task`, or '
                    'increase the value of `num_samples_per_task`.'.format(
                    test_shots, kwargs['num_samples_per_task']))
        else:
            shots = kwargs['num_samples_per_task'] // 2
    if test_shots is None:
        test_shots = shots

    dataset = Harmonic(num_samples_per_task=shots + test_shots, **kwargs)
    dataset = ClassSplitter(dataset, shuffle=shuffle,
        num_train_per_class=shots, num_test_per_class=test_shots)
    dataset.seed(seed)

    return dataset
