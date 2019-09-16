import os
import json

from torchmeta.datasets.cifar100.base import CIFAR100ClassDataset
from torchmeta.datasets.utils import get_asset
from torchmeta.utils.data import ClassDataset, CombinationMetaDataset


class CIFARFS(CombinationMetaDataset):
    """
    The CIFAR-FS dataset, introduced in [1]. This dataset contains
    images of 100 different classes from the CIFAR100 dataset [2].

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `cifar100` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to `N` in `N-way` 
        classification.

    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`, 
        `meta_val` and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g. `transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root 
        directory (under the `cifar100` folder). If the dataset is already 
        available, this does not download/process the dataset again.

    Notes
    -----
    The meta train/validation/test splits are over 64/16/20 classes from the
    CIFAR100 dataset.

    References
    ----------
    .. [1] Bertinetto L., Henriques J. F., Torr P. H.S., Vedaldi A. (2019).
           Meta-learning with differentiable closed-form solvers. In International
           Conference on Learning Representations (https://arxiv.org/abs/1805.08136)

    .. [2] Krizhevsky A. (2009). Learning Multiple Layers of Features from Tiny
           Images. (https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CIFARFSClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(CIFARFS, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class CIFARFSClassDataset(CIFAR100ClassDataset):
    subfolder = 'cifar-fs'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CIFARFSClassDataset, self).__init__(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)

    def download(self):
        if self._check_integrity():
            return
        super(CIFARFSClassDataset, self).download()

        subfolder = os.path.join(self.root, self.subfolder)
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)

        for split in ['train', 'val', 'test']:
            split_filename_labels = os.path.join(subfolder,
                self.filename_labels.format(split))
            if os.path.isfile(split_filename_labels):
                continue

            data = get_asset(self.folder, self.subfolder,
                '{0}.json'.format(split), dtype='json')
            with open(split_filename_labels, 'w') as f:
                json.dump(data, f)
