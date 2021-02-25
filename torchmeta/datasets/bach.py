import numpy as np
import os
import json
import h5py
from tqdm import tqdm

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import get_asset


class Bach(CombinationMetaDataset):
    """The Bach dataset """
    def __init__(self, root, num_classes_per_task=None, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False, process_features=True, min_num_samples_per_class=1):
        """
        Bach Choral Harmony dataset [1], [2]
        open-ml-id: 4552
        https://archive.ics.uci.edu/ml/datasets/Bach+Choral+Harmony

        Abstract: The data set is composed of 60 chorales (5665 events) by
        J.S. Bach (1675-1750). Each event of each chorale is labelled using 1
        among 101 chord labels and described through 14 features.

        Data Set Information:

        Pitch classes information has been extracted from MIDI sources downloaded
        from (JSB Chorales)[http://www.jsbchorales.net/]. Meter information has
        been computed through the Meter program which is part of the Melisma
        music analyser (Melisma)[http://www.link.cs.cmu.edu/music-analysis/].
        Chord labels have been manually annotated by a human expert.

        Attribute Information:

        1. Choral ID: corresponding to the file names from (Bach Central)[http://www.bachcentral.com/].
        2. Event number: index (starting from 1) of the event inside the chorale.
        3-14. Pitch classes: YES/NO depending on whether a given pitch is present.
        Pitch classes/attribute correspondence is as follows:
        C -> 3
        C#/Db -> 4
        D -> 5
        ...
        B -> 14
        15. Bass: Pitch class of the bass note
        16. Meter: integers from 1 to 5. Lower numbers denote less accented events,
        higher numbers denote more accented events.
        17. Chord label: Chord resonating during the given event.

        Notes
        ----------

        The features V1 and V2 are dropped during the processing. V1 is the Choral ID. V2 is
        the event number of the event inside the chorale.

        Parameters
        ----------
        root : string
            Root directory where the dataset folder `bach` exists.

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
            root directory (under the `bach' folder). If the dataset
            is already available, this does not download/process the dataset again.

        process_features : bool (default: `True`)
            If `True`, normalizes the numeric feature f according to (f-lower) / (upper - lower) where upper
            and lower are the min and max values of feature f of the meta-train dataset.
            And also one-hot encodes the categorical features.

        min_num_samples_per_class : int (default: 1)
            Minimal number of samples per class that need to be present for the class to be used.

        References
        -----

        [1] D. P. Radicioni and R. Esposito. Advances in Music Information Retrieval,
        chapter BREVE: an HMPerceptron-Based Chord Recognition System.
        Studies in Computational Intelligence,
        Zbigniew W. Ras and Alicja Wieczorkowska (Editors), Springer, 2010.

        [2] Esposito, R. and Radicioni, D. P., CarpeDiem: Optimizing the Viterbi
        Algorithm and Applications to Supervised Sequential Learning, Journal
        of Machine Learning Research, 10(Aug):1851-1880, 2009.
        """
        dataset = BachClassDataset(root,
                                   meta_train=meta_train,
                                   meta_val=meta_val,
                                   meta_test=meta_test,
                                   meta_split=meta_split,
                                   transform=transform,
                                   class_augmentations=class_augmentations,
                                   download=download,
                                   process_features=process_features,
                                   min_num_samples_per_class=min_num_samples_per_class)
        super(Bach, self).__init__(dataset,
                                   num_classes_per_task,
                                   target_transform=target_transform,
                                   dataset_transform=dataset_transform)


class BachClassDataset(ClassDataset):

    open_ml_id = 4552
    open_ml_url = 'https://www.openml.org/d/' + str(open_ml_id)
    dataset_name = "bach"

    folder = "bach"
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'
    filename_meta_data = 'meta_data.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split=None, transform=None,
                 class_augmentations=None, download=False, process_features=True, min_num_samples_per_class=None):
        super(BachClassDataset, self).__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                               meta_split=meta_split, class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root, self.filename_labels.format(self.meta_split))
        self.split_filename_meta_data = os.path.join(self.root, self.filename_meta_data)

        self._data_file = None
        self._data = None
        self._labels = None

        self._meta_data = None
        self._lower_upper = None
        self._categories = None

        if download:
            self.download(process_features, min_num_samples_per_class)

        if min_num_samples_per_class != self.meta_data["min_num_data_per_class"]:
            raise ValueError("min_num_samples_per_class given ({0}) does not match existing value"
                             "({1}).".format(min_num_samples_per_class, self.meta_data["min_num_data_per_class"]))

        if not self._check_integrity():
            raise RuntimeError('Bach integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        return BachDataset(index, data, label, transform=transform, target_transform=target_transform)

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

    @property
    def meta_data(self):
        if self._meta_data is None:
            with open(self.split_filename_meta_data, 'r') as f:
                self._meta_data = json.load(f)
        return self._meta_data

    @property
    def lower_upper(self):
        if self._lower_upper is None:
            self._lower_upper = {"lower": self.meta_data["lower"],
                                 "upper": self.meta_data["upper"],
                                 "feature_names_numerical":  self.meta_data["feature_names_numerical"]}
        return self._lower_upper

    @property
    def categories(self):
        if self._categories is None:
            self._categories = {"categories": self.meta_data["categories"],
                                "feature_names_categorical": self.meta_data["feature_names_categorical"]}
        return self._categories

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self, process_features, min_num_samples_per_class):

        if self._check_integrity():
            return

        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=self.open_ml_id)
        features = data.data
        targets = data.target
        feature_names = np.array(data.feature_names)

        # drop V1 and V2. V1 is the index of the choral, and V2 is the event number: index
        # (starting from 1) of the event inside the chorale.
        features_to_drop = np.array(['V1', 'V2'])
        idx_drop = [np.where(feature_names == v)[0][0] for v in features_to_drop]
        idx_keep = np.array([True] * feature_names.shape[0])
        for i in idx_drop:
            idx_keep[i] = False

        features = features[:, idx_keep]
        feature_names = feature_names[idx_keep]

        # get categorical feature names
        feature_names_cat = []
        for v in feature_names:
            if v in data.categories.keys():
                feature_names_cat.append(v)
        feature_names_cat = np.array(feature_names_cat)

        # get numerical feature names
        feature_names_num = []
        for fname in feature_names:
            if fname not in feature_names_cat:
                feature_names_num.append(fname)
        feature_names_num = np.array(feature_names_num)

        assert feature_names_num.shape[0] + feature_names_cat.shape[0] == len(feature_names)

        is_categorical = np.array([feature_name in feature_names_cat for feature_name in feature_names])
        is_numerical = np.array([feature_name in feature_names_num for feature_name in feature_names])

        # get categories
        categories = []
        for i in range(feature_names_cat.shape[0]):
            categories_i = np.unique(features[:, is_categorical][:, i])
            categories.append(categories_i.tolist())

        if process_features:
            from sklearn.preprocessing import OneHotEncoder
            ohe = OneHotEncoder(categories=categories, sparse=False, dtype=np.float)

        # for each meta-data-split, get the labels, then check which data-point belongs to the set (via a mask).
        # then, retrieve the features and targets belonging to the set. Then create hdf5 file for these features.
        for s, split in enumerate(['train', 'val', 'test']):
            targets_assets_split = get_asset(self.folder, '{0}.json'.format(split))

            is_in_split = [t in targets_assets_split for t in targets]
            features_split = features[is_in_split, :]
            targets_split = targets[is_in_split]
            assert targets_split.shape[0] == features_split.shape[0]

            unique_targets_split = np.unique(targets_split)

            # first we check how many data-points are associated with each class. If it is less than the threshold,
            # min_num_samples_per_class, then we discard the whole class.
            num_dat_per_class = []
            for label in unique_targets_split:
                num_dat_per_class.append(features_split[targets_split == label, :].shape[0])
            num_dat_per_class = np.array(num_dat_per_class)

            # remove labels which have less data-points associate with them than the threshold min_num_data_per_class.
            classes_to_keep = num_dat_per_class >= min_num_samples_per_class
            unique_targets_with_enough_data_split = unique_targets_split[classes_to_keep]

            if unique_targets_with_enough_data_split.shape[0] < unique_targets_split.shape[0]:
                print("split: ({2}): number of unique targets with enough data ({0}) is smaller than "
                      "number of unique targets in assets ({1})".format(
                    unique_targets_with_enough_data_split.shape[0], unique_targets_split.shape[0], split))

            # write unique targets to json file.
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(unique_targets_with_enough_data_split.tolist(), f)

            # get pre-processing stats from the meta-train split
            if split == 'train':
                # numerical features
                lower, upper = np.zeros(features.shape[1]), np.ones(features.shape[1])
                if process_features:
                    # lower upper
                    lower = np.min(features[:, is_numerical], axis=0)
                    upper = np.max(features[:, is_numerical], axis=0)
                self._lower_upper = {'lower': lower.tolist(), 'upper': upper.tolist(),
                                     'features_names': feature_names_num.tolist()}

            # apply pre-processing of features
            if process_features:
                features_split_num = np.true_divide((features_split[:, is_numerical] - lower), (upper - lower))
                features_split_cat = ohe.fit_transform(features_split[:, is_categorical])
                features_split = np.hstack([features_split_num, features_split_cat])

            # write data (features and class labels)
            filename = os.path.join(self.root, self.filename.format(split))
            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')

                for i, label in enumerate(tqdm(unique_targets_with_enough_data_split, desc=filename)):
                    data_class = features_split[targets_split == label, :]
                    group.create_dataset(label, data=data_class)

        # store meta-data of the dataset (not the meta-dataset).
        # Extend this dictionary if you want to store more meta-data of the meta-dataset.
        meta_data = {"min_num_data_per_class": min_num_samples_per_class,
                     "lower": lower.tolist(),
                     "upper": upper.tolist(),
                     "feature_names_numerical": feature_names_num.tolist(),
                     "feature_names_categorical": feature_names_cat.tolist(),
                     "categories": categories,
                     "dropped_features": features_to_drop.tolist()}

        with open(self.split_filename_meta_data, 'w') as f:
            json.dump(meta_data, f)


class BachDataset(Dataset):
    def __init__(self, index, data, label, transform=None, target_transform=None):
        super(BachDataset, self).__init__(index, transform=transform, target_transform=target_transform)
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


def create_asset(root='data', fractions=None, seed=42):
    """This methods creates the assets of the Bach dataset. These are the meta-dataset splits from the
    original data. Only run this method in case you want to create new assets. Once created, copy the assets to
    this directory: torchmeta.datasets.assets.bach. You can also manually change the assets."""

    # split fractions: train, valid, test
    if fractions is None:
        fractions = [0.6, 0.2, 0.2]
    assert sum(fractions) == 1

    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=BachClassDataset.open_ml_id)
    unique_targets = np.unique(data.target)
    num_unique_targets = len(unique_targets)

    num_split = [int(f * num_unique_targets) for f in fractions]
    num_split[2] = num_unique_targets - num_split[0] - num_split[1]
    assert sum(num_split) == num_unique_targets

    # split unique labels randomly
    np.random.seed(seed)
    perm = np.random.permutation(num_unique_targets)
    targets_split = {'train': [unique_targets[i] for i in perm[:num_split[0]]],
                     'val': [unique_targets[i] for i in perm[num_split[0]: num_split[0] + num_split[1]]],
                     'test': [unique_targets[i] for i in perm[num_split[0] + num_split[1]:]]}

    # write splits
    root_path = os.path.join(os.path.expanduser(root), BachClassDataset.folder)
    for split in ["train", "val", "test"]:
        asset_filename = os.path.join(root_path, "{0}.json".format(split))
        with open(asset_filename, 'w') as f:
            json.dump(targets_split[split], f)

