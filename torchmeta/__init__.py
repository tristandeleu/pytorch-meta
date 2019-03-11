from torchmeta import datasets
from torchmeta import transforms

from .dataloaders import MetaDataLoader, BasicMetaDataLoader

import warnings

def fractions_to_lengths(*args, **kwargs):
    warnings.warn('The function `torchmeta.fractions_to_lengths` is deprecated.',
        DeprecationWarning, stacklevel=2)
    from .splitters import fractions_to_lengths as fractions_to_lengths_
    return fractions_to_lengths_(*args, **kwargs)

def classwise_split(*args, **kwargs):
    warnings.warn('The function `torchmeta.classwise_split` is deprecated. '
        'Please use `torchmeta.tasks.ConcatTask` to create a task from '
        'multiple individual class datasets. If you are using `torchmeta.'
        'classwise_split` together with `torchmeta.stratified_split`, you can'
        'use `torchmeta.transforms.ClassSplitter`.', DeprecationWarning, stacklevel=2)
    from .splitters import classwise_split as classwise_split_
    return classwise_split_(*args, **kwargs)

def stratified_split(*args, **kwargs):
    warnings.warn('The function `torchmeta.stratified_split` is deprecated. '
        'Please use the dataset transformation `torchmeta.transforms.'
        'ClassSplitter` instead.', DeprecationWarning, stacklevel=2)
    from .splitters import stratified_split as stratified_split_
    return stratified_split_(*args, **kwargs)
