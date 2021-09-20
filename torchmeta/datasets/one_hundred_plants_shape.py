import numpy as np
import os
import json
import h5py
from tqdm import tqdm

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import get_asset


class PlantsShape(CombinationMetaDataset):
    """The PlantsShape dataset """
    def __init__(self, root, num_classes_per_task=None, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False, process_features=False):
        """
        One-hundred plant species leaves dataset (Class = Shape) [1], [2], [3]
        open-ml-id: 1492
        https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set) - 2010


           (a) Original owners of colour Leaves Samples:

         James Cope, Thibaut Beghin, Paolo Remagnino, Sarah Barman.
         The colour images are not included.
         The Leaves were collected in the Royal Botanic Gardens, Kew, UK.
         email: james.cope@kingston.ac.uk

           (b) This dataset consists of work carried out by James Cope, Charles Mallah, and James Orwell.
         Donor of database Charles Mallah: charles.mallah@kingston.ac.uk; James Cope:  james.cope@kingston.ac.uk

        The original data directory contains the binary images (masks) of the leaf samples (colour images not included).
        There are three features for each image: Shape, Margin and Texture.
        For each feature, a 64 element vector is given per leaf sample.
        These vectors are taken as a contiguous descriptor (for shape) or histograms (for texture and margin).
        So, there are three different files, one for each feature problem:
         * 'data_Sha_64.txt' -> prediction based on shape [dataset provided here]
         * 'data_Tex_64.txt' -> prediction based on texture
         * 'data_Mar_64.txt' -> prediction based on margin

        Each row has a 64-element feature vector followed by the Class label.
        There is a total of 1600 samples with 16 samples per leaf class (100 classes), and no missing values.

        Three 64 element feature vectors per sample.

        Parameters
        ----------
        root : string
            Root directory where the dataset folder `one_hundred_plants_shape` exists.

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
            root directory (under the `one_hundred_plants_shape' folder). If the dataset
            is already available, this does not download/process the dataset again.

        process_features : bool (default: `False`)
            If `True`, normalizes each feature f according to (f-mean) / (std + 1e-10) where
            mean and std are the mean and standard deviation of the feature f of the meta-train dataset.

        References
        -----
        [1] Charles Mallah, James Cope, James Orwell.
        Plant Leaf Classification Using Probabilistic Integration of Shape, Texture and Margin Features.
        Signal Processing, Pattern Recognition and Applications, in press.

        [2] J. Cope, P. Remagnino, S. Barman, and P. Wilkin.
        Plant texture classification using gabor co-occurrences.
        Advances in Visual Computing, pages 699-677, 2010.

        [3] T. Beghin, J. Cope, P. Remagnino, and S. Barman.
        Shape and texture based plant leaf classification.
        In: Advanced Concepts for Intelligent Vision Systems, pages 345-353. Springer, 2010.

        """
        dataset = PlantsShapeClassDataset(root,
                                          meta_train=meta_train,
                                          meta_val=meta_val,
                                          meta_test=meta_test,
                                          meta_split=meta_split,
                                          transform=transform,
                                          class_augmentations=class_augmentations,
                                          download=download,
                                          normalize=process_features)
        super(PlantsShape, self).__init__(dataset,
                                            num_classes_per_task,
                                            target_transform=target_transform,
                                            dataset_transform=dataset_transform)


class PlantsShapeClassDataset(ClassDataset):

    open_ml_id = 1492
    open_ml_url = 'https://www.openml.org/d/' + str(open_ml_id)
    dataset_name = "one_hundred_plants_shape"

    folder = "one_hundred_plants_shape"
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'
    filename_mean_std = 'features_mean_std.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split=None, transform=None,
                 class_augmentations=None, download=False, normalize=True):
        super(PlantsShapeClassDataset, self).__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                                      meta_split=meta_split, class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root, self.filename_labels.format(self.meta_split))
        self.split_filename_mean_std = os.path.join(self.root, self.filename_mean_std)

        self._data_file = None
        self._data = None
        self._labels = None
        self._mean_std = None

        if download:
            self.download(normalize)

        if not self._check_integrity():
            raise RuntimeError('PlantsShape integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        return PlantsShapeDataset(index, data, label, transform=transform, target_transform=target_transform)

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
    def mean_std(self):
        if self._mean_std is None:
            with open(self.split_filename_mean_std, 'r') as f:
                self._mean_std = json.load(f)
        return self._mean_std

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self, normalize):

        if self._check_integrity():
            return

        from sklearn.datasets import fetch_openml

        data = fetch_openml(data_id=self.open_ml_id)
        features = data.data
        targets = data.target

        os.makedirs(self.root, exist_ok=True)

        # for each meta-data-split, get the labels, then check which data-point belongs to the set (via a mask).
        # then, retrieve the features and targets belonging to the set. Then create hdf5 file for these features.
        for s, split in enumerate(['train', 'val', 'test']):
            targets_assets_split = get_asset(self.folder, '{0}.json'.format(split))

            is_in_split = [t in targets_assets_split for t in targets]
            features_split = features.loc[is_in_split]
            targets_split = targets.loc[is_in_split]
            assert targets_split.shape[0] == features_split.shape[0]

            unique_targets_split = np.sort(np.unique(targets_split))
            if len(targets_assets_split) > unique_targets_split.shape[0]:
                print(f"unique set of labels ({(unique_targets_split.shape[0])}) is smaller than set of labels "
                      f"given by assets ({len(targets_assets_split)}). Proceeding with unique set of labels.")

            # write unique targets to json file.
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(unique_targets_split.tolist(), f)

            # normalize to zero mean and standard deviation 1 with stats from 'train' split only
            if split == 'train':
                mean, std = np.zeros(features.shape[1]), np.ones(features.shape[1])
                if normalize:
                    mean = np.mean(features_split, axis=0)
                    std = np.std(features_split, axis=0)

                self._mean_std = {'mean': mean.tolist(), 'std': std.tolist()}
                mean_std_filename = os.path.join(self.root, self.filename_mean_std)
                with open(mean_std_filename, 'w') as f:
                    json.dump(self._mean_std, f)

            mean_std = self.mean_std
            mean = np.array(mean_std['mean'])
            std = np.array(mean_std['std'])
            features_split = (features_split - mean) / (std + 1e-10)

            # write data (features and class labels)
            filename = os.path.join(self.root, self.filename.format(split))
            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')

                for i, label in enumerate(tqdm(unique_targets_split, desc=filename)):
                    data_class = features_split.loc[targets_split == label]
                    group.create_dataset(label, data=data_class)


class PlantsShapeDataset(Dataset):
    def __init__(self, index, data, label, transform=None, target_transform=None):
        super(PlantsShapeDataset, self).__init__(index, transform=transform, target_transform=target_transform)
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
    """This methods creates the assets of the PlantsShape dataset. These are the meta-dataset splits from the
    original data. Only run this method in case you want to create new assets. Once created, copy the assets to
    this directory: torchmeta.datasets.assets.one_hundred_plants_shape. You can also manually change the assets."""

    # split fractions: train, valid, test
    if fractions is None:
        fractions = [0.7, 0.15, 0.15]
    assert sum(fractions) == 1

    from sklearn.datasets import fetch_openml

    data = fetch_openml(data_id=PlantsShapeClassDataset.open_ml_id)
    unique_targets = np.unique(data.target)
    num_unique_targets = len(unique_targets)

    num_split = [int(f * num_unique_targets) for f in fractions]
    num_split[1] = num_unique_targets - num_split[0] - num_split[2]
    assert sum(num_split) == num_unique_targets

    # split unique labels randomly
    np.random.seed(seed)
    perm = np.random.permutation(num_unique_targets)
    targets_split = {'train': [unique_targets[i] for i in perm[:num_split[0]]],
                     'val': [unique_targets[i] for i in perm[num_split[0]: num_split[0] + num_split[1]]],
                     'test': [unique_targets[i] for i in perm[num_split[0] + num_split[1]:]]}

    # write splits
    root_path = os.path.join(os.path.expanduser(root), PlantsShapeClassDataset.folder)
    for split in ["train", "val", "test"]:
        asset_filename = os.path.join(root_path, "{0}.json".format(split))
        with open(asset_filename, 'w') as f:
            json.dump(targets_split[split], f)
