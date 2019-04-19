import numpy as np
import os
import json
import glob
import h5py
from PIL import Image, ImageOps

from torch.utils.data import Dataset
from torchmeta.dataset import ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import list_dir, download_url
from torchmeta.datasets.utils import get_asset

class Omniglot(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, use_vinyals_split=True,
                 transform=None, target_transform=None, dataset_transform=None,
                 meta_split=None, class_augmentations=None, download=False):
        dataset = OmniglotClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            use_vinyals_split=use_vinyals_split, transform=transform,
            target_transform=target_transform, meta_split=meta_split,
            class_augmentations=class_augmentations, download=download)
        super(Omniglot, self).__init__(dataset, num_classes_per_task,
            dataset_transform=dataset_transform)


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
                 use_vinyals_split=True, transform=None, target_transform=None,
                 class_augmentations=None, meta_split=None, download=False):
        super(OmniglotClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        if self.meta_val and (not use_vinyals_split):
            raise ValueError()

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.use_vinyals_split = use_vinyals_split
        self.transform = transform
        self.target_transform = target_transform

        self.split_filename = os.path.join(self.root, self.filename)
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format('vinyals_' if use_vinyals_split else '',
            self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError()
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        character_name = '/'.join(self.labels[index])
        data = self.data[character_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index, self.target_transform)

        return OmniglotDataset(data, character_name, transform=transform,
            target_transform=target_transform)

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
                labels_filename = os.path.join(self.root, self.filename_labels.format('', split))
                with open(labels_filename, 'w') as f_labels:
                    labels = sorted(characters)
                    json.dump(labels, f_labels)

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name, alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet, character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                shutil.rmtree(os.path.join(self.root, name))

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename_labels.format('vinyals_', split))
            data = get_asset(self.folder, '{0}.json'.format(split), dtype='json')

            with open(filename, 'w') as f:
                labels = sorted([('images_{0}'.format(name), alphabet, character) for (name, alphabets)
                    in data.items() for (alphabet, characters) in alphabets.items()
                    for character in characters])
                json.dump(labels, f)


class OmniglotDataset(Dataset):
    def __init__(self, data, character_name, transform=None, target_transform=None):
        super(OmniglotDataset, self).__init__()
        self.data = data
        self.character_name = character_name
        self.transform = transform
        self.target_transform = target_transform

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
