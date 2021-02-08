import numpy as np
import os
import json
import h5py
from tqdm import tqdm

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import get_asset


class Covertype(CombinationMetaDataset):
    """The Covertype dataset """
    def __init__(self, root, num_classes_per_task=None, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        """
        Covertype [1], [2], [3]:

        The dataset is loaded and processed with benchlib. Originally it is from open-ml.
        https://www.openml.org/d/1596
        https://archive.ics.uci.edu/ml/datasets/covertype

        Predicting forest cover type from cartographic variables only (no remotely
        sensed data). The actual forest cover type for a given observation
        (30 x 30 meter cell) was determined from US Forest Service (USFS) Region
        2 Resource Information System (RIS) data. Independent variables were derived
        from data originally obtained from US Geological Survey (USGS) and USFS
        data. Data is in raw form (not scaled) and contains binary (0 or 1) columns
        of data for qualitative independent variables (wilderness areas and soil types).

        This study area includes four wilderness areas located in the Roosevelt National
        Forest of northern Colorado. These areas represent forests with minimal
        human-caused disturbances, so that existing forest cover types are more a
        result of ecological processes rather than forest management practices.

        Some background information for these four wilderness areas: Neota (area 2)
        probably has the highest mean elevational value of the 4 wilderness areas.
        Rawah (area 1) and Comanche Peak (area 3) would have a lower mean elevational
        value, while Cache la Poudre (area 4) would have the lowest mean elevational value.

        As for primary major tree species in these areas, Neota would have
        spruce/fir (type 1), while Rawah and Comanche Peak would probably have lodgepole
        pine (type 2) as their primary species, followed by spruce/fir and aspen (type 5).
        Cache la Poudre would tend to have Ponderosa pine (type 3), Douglas-fir (type 6),
        and cottonwood/willow (type 4).

        The Rawah and Comanche Peak areas would tend to be more typical of the overall
        dataset than either the Neota or Cache la Poudre, due to their assortment of tree
        species and range of predictive variable values (elevation, etc.) Cache la Poudre
        would probably be more unique than the others, due to its relatively low elevation
        range and species composition.

        Parameters
        ----------
        root : string
            Root directory where the dataset folder `covertype_task_id_2118` exists.

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
            root directory (under the `covertype_task_id_2118` folder). If the dataset
            is already available, this does not download/process the dataset again.

        References
        -----
        [1] Blackard, Jock A. and Denis J. Dean. 2000. "Comparative Accuracies of Artificial
        Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from
        Cartographic Variables." Computers and Electronics in Agriculture 24(3):131-151.

        [2] Blackard, Jock A. and Denis J. Dean. 1998. "Comparative Accuracies of Neural
        Networks and Discriminant Analysis in Predicting Forest Cover Types from
        Cartographic Variables." Second Southern Forestry GIS Conference. University of
        Georgia. Athens, GA. Pages 189-199.

        [3] Blackard, Jock A. 1998. "Comparison of Neural Networks and Discriminant Analysis
        in Predicting Forest Cover Types." Ph.D. dissertation. Department of Forest Sciences.
        Colorado State University. Fort Collins, Colorado. 165 pages.
        """
        dataset = CovertypeClassDataset(root,
                                        meta_train=meta_train,
                                        meta_val=meta_val,
                                        meta_test=meta_test,
                                        meta_split=meta_split,
                                        transform=transform,
                                        class_augmentations=class_augmentations,
                                        download=download)
        super(Covertype, self).__init__(dataset,
                                        num_classes_per_task,
                                        target_transform=target_transform,
                                        dataset_transform=dataset_transform)


class CovertypeClassDataset(ClassDataset):

    benchlib_namespace = "openml_datasets"
    benchlib_dataset_name = "covertype_task_id_2118"

    folder = "covertype_task_id_2118"
    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split=None, transform=None,
                 class_augmentations=None, download=False):
        super(CovertypeClassDataset, self).__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
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
            raise RuntimeError('Covertype integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CovertypeDataset(index, data, label, transform=transform, target_transform=target_transform)

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


class CovertypeDataset(Dataset):
    def __init__(self, index, data, label, transform=None, target_transform=None):
        super(CovertypeDataset, self).__init__(index, transform=transform, target_transform=target_transform)
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


def create_asset(root='data', labels=None):
    """This methods creates the assets of the covertype dataset. These are the meta-dataset splits from the
    original data. Only run this method in case you want to create new assets. Once created, copy the assets to
    this directory: torchmeta.datasets.assets.covertype_task_id_2118. You can also manually change the assets."""
    if labels is None:
        # there are 7 classes: "0", "1", "2", ... "6".
        # these are the default splits.
        labels = {"train": ["1", "3", "6"],
                  "val": ["0", "2"],
                  "test": ["4", "5"]}

    root_path = os.path.join(os.path.expanduser(root), CovertypeClassDataset.folder)
    print(root_path)

    for split in ["train", "val", "test"]:
        asset_filename = os.path.join(root_path, "{0}.json".format(split))
        with open(asset_filename, 'w') as f:
            json.dump(labels[split], f)
