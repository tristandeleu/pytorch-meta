import os
import json
import glob
import h5py
from PIL import Image, ImageOps

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import list_dir, download_url
from torchmeta.datasets.utils import get_asset


class Omniglot(CombinationMetaDataset):
    """
    The Omniglot dataset [1]. A dataset of 1623 handwritten characters from 
    50 different alphabets. 

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `omniglot` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way" 
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

    use_vinyals_split : bool (default: `True`)
        If set to `True`, the dataset uses the splits defined in [3]. If `False`, 
        then the meta-train split corresponds to `images_background`, and the 
        meta-test split corresponds to `images_evaluation` (raises an error when 
        calling the meta-validation split).

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the zip files and processes the dataset in the root 
        directory (under the `omniglot` folder). If the dataset is already 
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from the original [Omniglot repository]
    (https://github.com/brendenlake/omniglot). The meta train/validation/test 
    splits used in [3] are taken from [this repository]
    (https://github.com/jakesnell/prototypical-networks). These splits are 
    over 1028/172/423 classes (characters).

    References
    ----------
    .. [1] Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level 
           concept learning through probabilistic program induction. Science, 350(6266), 
           1332-1338 (http://www.sciencemag.org/content/350/6266/1332.short)

    .. [2] Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2019). The Omniglot 
           Challenge: A 3-Year Progress Report (https://arxiv.org/abs/1902.03477)

    .. [3] Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016). 
           Matching Networks for One Shot Learning. In Advances in Neural 
           Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 use_vinyals_split=True, transform=None, target_transform=None,
                 dataset_transform=None, class_augmentations=None, download=False):
        dataset = OmniglotClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            use_vinyals_split=use_vinyals_split, transform=transform,
            meta_split=meta_split, class_augmentations=class_augmentations,
            download=download)
        super(Omniglot, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class OmniglotClassDataset(ClassDataset):
    folder = 'omniglot'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}{1}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, use_vinyals_split=True, transform=None,
                 class_augmentations=None, download=False):
        super(OmniglotClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        if self.meta_val and (not use_vinyals_split):
            raise ValueError('Trying to use the meta-validation without the '
                'Vinyals split. You must set `use_vinyals_split=True` to use '
                'the meta-validation split.')

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename)
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format('vinyals_' if use_vinyals_split else '',
            self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Omniglot integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        character_name = '/'.join(self.labels[index % self.num_classes])
        data = self.data[character_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return OmniglotDataset(index, data, character_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
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
        import zipfile
        import shutil

        if self._check_integrity():
            return

        for name in self.zips_md5:
            zip_filename = '{0}.zip'.format(name)
            filename = os.path.join(self.root, zip_filename)
            if os.path.isfile(filename):
                continue

            url = '{0}/{1}'.format(self.download_url_prefix, zip_filename)
            download_url(url, self.root, zip_filename, self.zips_md5[name])

            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.root)

        filename = os.path.join(self.root, self.filename)
        with h5py.File(filename, 'w') as f:
            for name in self.zips_md5:
                group = f.create_group(name)

                alphabets = list_dir(os.path.join(self.root, name))
                characters = [(name, alphabet, character) for alphabet in alphabets
                    for character in list_dir(os.path.join(self.root, name, alphabet))]

                split = 'train' if name == 'images_background' else 'test'
                labels_filename = os.path.join(self.root,
                    self.filename_labels.format('', split))
                with open(labels_filename, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name,
                        alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                        character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                shutil.rmtree(os.path.join(self.root, name))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename_labels.format(
                'vinyals_', split))
            data = get_asset(self.folder, '{0}.json'.format(split), dtype='json')

            with open(filename, 'w') as f:
                labels = sorted([('images_{0}'.format(name), alphabet, character)
                    for (name, alphabets) in data.items()
                    for (alphabet, characters) in alphabets.items()
                    for character in characters])
                json.dump(labels, f)


class OmniglotDataset(Dataset):
    def __init__(self, index, data, character_name,
                 transform=None, target_transform=None):
        super(OmniglotDataset, self).__init__(index, transform=transform,
                                              target_transform=target_transform)
        self.data = data
        self.character_name = character_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.character_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
