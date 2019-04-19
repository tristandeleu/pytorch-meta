from torchmeta.transforms.categorical import CategoricalWrapper, FixedCategory
from torchmeta.transforms.augmentations import Rotation, HorizontalFlip, VerticalFlip
from torchmeta.transforms.splitters import ClassSplitter, WeightedClassSplitter

def CategoricalTaskTarget():
    import warnings
    warnings.warn('The dataset transform `CategoricalTaskTarget` is deprecated '
        'and will be removed. You can use `torchmeta.transforms.'
        'CategoricalWrapper()` instead.', stacklevel=2)
    return CategoricalWrapper()
