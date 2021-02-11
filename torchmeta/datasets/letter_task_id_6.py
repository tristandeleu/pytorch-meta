import numpy as np
import os
import json
import h5py
from tqdm import tqdm

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import get_asset


class Letter(CombinationMetaDataset):
    """The Letter Image Recognition Dataset """
    def __init__(self, root, num_classes_per_task=None, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        """
        Letter Image Recognition Data [1]:

        https://archive.ics.uci.edu/ml/datasets/Letter+Recognition - 01-01-1991

        The objective is to identify each of a large number of black-and-white
        rectangular pixel displays as one of the 26 capital letters in the English
        alphabet.  The character images were based on 20 different fonts and each
        letter within these 20 fonts was randomly distorted to produce a file of
        20,000 unique stimuli.  Each stimulus was converted into 16 primitive
        numerical attributes (statistical moments and edge counts) which were then
        scaled to fit into a range of integer values from 0 through 15.  We
        typically train on the first 16000 items and then use the resulting model
        to predict the letter category for the remaining 4000.  See the article
        cited above for more details.

        The dataset is loaded and processed with benchlib. Originally it is from open-ml.
        https://www.openml.org/d/6

        Parameters
        ----------
        root : string
            Root directory where the dataset folder `letter_task_id_6` exists.

        num_classes_per_task : int
            Number of classes per tasks. This corresponds to "N" in "N-way"
            classification.

        meta_train : bool (default: `False`)
            Use the meta-train split of the dataset. If set to `True`, then the
            arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
            of these three arguments must be set to `True`.

        meta_val : bool (default: `False`)
            Use the meta-validation split of the dataset. If set to `True`, then the
            arguments `meta_train` and `meta_test` must be set to `False`. Exactly
            one of these three arguments must be set to `True`.

        meta_test : bool (default: `False`)
            Use the meta-test split of the dataset. If set to `True`, then the
            arguments `meta_train` and `meta_val` must be set to `False`. Exactly
            one of these three arguments must be set to `True`.

        meta_split : string in {'train', 'val', 'test'}, optional
            Name of the split to use. This overrides the arguments `meta_train`,
            `meta_val` and `meta_test` if all three are set to `False`.

        transform : callable, optional
            A function/transform that takes a numpy array or a pytorch array
            (depending when the transforms is applied), and returns a transformed
            version.

        target_transform : callable, optional
            A function/transform that takes a target, and returns a transformed
            version.

        dataset_transform : callable, optional
            A function/transform that takes a dataset (ie. a task), and returns a
            transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

        class_augmentations : list of callable, optional
            A list of functions that augment the dataset with new classes. These
            classes are transformations of existing classes.

        download : bool (default: `False`)
            If `True`, downloads the original files and processes the dataset in the
            root directory (under the `letter_task_id_6` folder). If the dataset
            is already available, this does not download/process the dataset again.

        References
        -----
        [1] P. W. Frey and D. J. Slate. "Letter Recognition Using Holland-style
        Adaptive Classifiers". Machine Learning 6(2), 1991
        """
        dataset = LetterClassDataset(root,
                                     meta_train=meta_train,
                                     meta_val=meta_val,
                                     meta_test=meta_test,
                                     meta_split=meta_split,
                                     transform=transform,
                                     class_augmentations=class_augmentations,
                                     download=download)
        super(Letter, self).__init__(dataset,
                                     num_classes_per_task,
                                     target_transform=target_transform,
                                     dataset_transform=dataset_transform)


class LetterClassDataset(ClassDataset):

    benchlib_namespace = "openml_datasets"
    benchlib_dataset_name = "letter_task_id_6"

    folder = "letter_task_id_6"
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split=None, transform=None,
                 class_augmentations=None, download=False):
        super(LetterClassDataset, self).__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                                 meta_split=meta_split, class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root, self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Letter integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return LetterDataset(index, data, label, transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):

        if self._check_integrity():
            return

        from benchlib.datasets.syne_datasets import get_syne_dataset
        from benchlib.datasets.data_detergent import DataDetergent

        # feature transforms are performed by the DataDetergent
        d = DataDetergent(get_syne_dataset(namespace=self.benchlib_namespace,
                                           dataset_name=self.benchlib_dataset_name + "/"),
                          do_impute_nans=True,
                          do_normalize_cols=True,
                          do_remove_const_features=True)

        # stack the features and targets into one big numpy array each, since we want a new split.
        features = []
        targets = []
        for split in ['train', 'val', 'test']:
            if split == 'train':
                data = d.get_training_data()
            elif split == 'val':
                data = d.get_validation_data()
            elif split == 'test':
                data = d.get_test_data()
            else:
                raise ValueError(f"split {split} not found.")
            features.append(data[0])
            targets.append(data[1])
        data = None
        features = np.concatenate(features, axis=0)
        targets = np.concatenate(targets, axis=0)

        # for each meta-data-split, get the labels, then check which data-point belongs to the set (via a mask).
        # then, retrieve the features and targets belonging to the set. Then create hdf5 file for these features.
        for s, split in enumerate(['train', 'val', 'test']):
            label_set = get_asset(self.folder, '{0}.json'.format(split))
            label_set_integers = [int(l) for l in label_set]

            is_in_set = [t in label_set_integers for t in targets]
            features_set = features[is_in_set, :]
            targets_set = targets[is_in_set]
            assert targets_set.shape[0] == features_set.shape[0]

            unique_targets_set = np.sort(np.unique(targets_set))
            if len(label_set_integers) > unique_targets_set.shape[0]:
                print(f"unique set of labels is smaller ({len(unique_targets_set.shape[0])}) than set of labels "
                      f"given by assets ({len(label_set_integers)}). Proceeding with unique set of labels.")

            # write unique targets with enough data to json file. this is not necessarily the same as the tag set
            len_str = int(np.ceil(np.log10(unique_targets_set.shape[0] + 1)))
            unique_targets_str = [str(i).zfill(len_str) for i in unique_targets_set]

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(unique_targets_str, f)

            # write data (features and class labels)
            filename = os.path.join(self.root, self.filename.format(split))
            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.float64)

                for i, label in enumerate(tqdm(unique_targets_str, desc=filename)):
                    data_class = features_set[targets_set == int(label), :]
                    group.create_dataset(label, data=data_class)  # , dtype=dtype)


class LetterDataset(Dataset):
    def __init__(self, index, data, label, transform=None, target_transform=None):
        super(LetterDataset, self).__init__(index, transform=transform, target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        features = self.data[index, :]
        target = self.label

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return features, target


def create_asset(root='data', number_of_classes_per_split=None, numpy_seed=42):
    """This methods creates the assets of the letter dataset. These are the meta-dataset splits from the
    original data. Only run this method in case you want to create new assets. Once created, copy the assets to
    this directory: torchmeta.datasets.assets.letter_task_id_6. You can also manually change the assets."""

    # split fractions: train, valid, tes
    if number_of_classes_per_split is None:
        number_of_classes_per_split = {"train": 15,
                                       "val": 5,
                                       "test": 6}
    num_classes = 0
    for key in number_of_classes_per_split:
        num_classes += number_of_classes_per_split[key]
    assert num_classes == 26

    def make_split(num_classes, number_of_classes_per_split):
        """get permutation of labels and split according to number of classes per split"""
        np.random.seed(numpy_seed)

        perm = np.random.permutation(num_classes)
        class_splits = {}
        start = 0
        for split in ["train", "val", "test"]:
            num_c = number_of_classes_per_split[split]

            class_splits[split] = [str(i) for i in perm[start:start+num_c]]
            start += num_c
        return class_splits

    # Split the classes according to the number of classes per split, and store the splits in the data directory.
    class_splits = make_split(num_classes, number_of_classes_per_split)
    print(class_splits)
    root_path = os.path.join(os.path.expanduser(root), LetterClassDataset.folder)
    for split in ["train", "val", "test"]:
        asset_filename = os.path.join(root_path, "{0}.json".format(split))
        with open(asset_filename, 'w') as f:
            json.dump(class_splits[split], f)