import numpy as np
import os
import json
import h5py
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_url
from torchmeta.utils.data import Dataset, ClassDataset


class CIFAR100ClassDataset(ClassDataset):
    folder = 'cifar100'
    subfolder = None
    download_url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    gz_folder = 'cifar-100-python'
    gz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    files_md5 = {
        'train': '16019d7e3df5f24257cddd939b257f8d',
        'test': 'f0ef6b0ae62326f3e7ffdfab6717acfc',
        'meta': '7973b15100ade9c7d40fb424638fde48'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}_labels.json'
    filename_fine_names = 'fine_names.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CIFAR100ClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        if self.subfolder is None:
            raise ValueError()

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename_labels = os.path.join(self.root, self.subfolder,
            self.filename_labels.format(self.meta_split))
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('CIFAR100 integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        coarse_label_name, fine_label_name = self.labels[index % self.num_classes]
        data = self.data['{0}/{1}'.format(coarse_label_name, fine_label_name)]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CIFAR100Dataset(index, data, coarse_label_name, fine_label_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(os.path.join(self.root, self.filename), 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (self._check_integrity_data()
            and os.path.isfile(self.split_filename_labels)
            and os.path.isfile(os.path.join(self.root, self.filename_fine_names)))

    def _check_integrity_data(self):
        return os.path.isfile(os.path.join(self.root, self.filename))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import tarfile
        import pickle
        import shutil

        if self._check_integrity_data():
            return

        gz_filename = '{0}.tar.gz'.format(self.gz_folder)
        download_url(self.download_url, self.root, filename=gz_filename,
                     md5=self.gz_md5)
        with tarfile.open(os.path.join(self.root, gz_filename), 'r:gz') as tar:
            tar.extractall(path=self.root)

        train_filename = os.path.join(self.root, self.gz_folder, 'train')
        check_integrity(train_filename, self.files_md5['train'])
        with open(train_filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            images = data[b'data']
            fine_labels = data[b'fine_labels']
            coarse_labels = data[b'coarse_labels']

        test_filename = os.path.join(self.root, self.gz_folder, 'test')
        check_integrity(test_filename, self.files_md5['test'])
        with open(test_filename, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            images = np.concatenate((images, data[b'data']), axis=0)
            fine_labels = np.concatenate((fine_labels, data[b'fine_labels']), axis=0)
            coarse_labels = np.concatenate((coarse_labels, data[b'coarse_labels']), axis=0)

        images = images.reshape((-1, 3, 32, 32))
        images = images.transpose((0, 2, 3, 1))

        meta_filename = os.path.join(self.root, self.gz_folder, 'meta')
        check_integrity(meta_filename, self.files_md5['meta'])
        with open(meta_filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            fine_label_names = data['fine_label_names']
            coarse_label_names = data['coarse_label_names']

        filename = os.path.join(self.root, self.filename)
        fine_names = dict()
        with h5py.File(filename, 'w') as f:
            for i, coarse_name in enumerate(coarse_label_names):
                group = f.create_group(coarse_name)
                fine_indices = np.unique(fine_labels[coarse_labels == i])
                for j in fine_indices:
                    dataset = group.create_dataset(fine_label_names[j],
                        data=images[fine_labels == j])
                fine_names[coarse_name] = [fine_label_names[j] for j in fine_indices]

        filename_fine_names = os.path.join(self.root, self.filename_fine_names)
        with open(filename_fine_names, 'w') as f:
            json.dump(fine_names, f)

        gz_folder = os.path.join(self.root, self.gz_folder)
        if os.path.isdir(gz_folder):
            shutil.rmtree(gz_folder)
        if os.path.isfile('{0}.tar.gz'.format(gz_folder)):
            os.remove('{0}.tar.gz'.format(gz_folder))


class CIFAR100Dataset(Dataset):
    def __init__(self, index, data, coarse_label_name, fine_label_name,
                 transform=None, target_transform=None):
        super(CIFAR100Dataset, self).__init__(index, transform=transform,
                                              target_transform=target_transform)
        self.data = data
        self.coarse_label_name = coarse_label_name
        self.fine_label_name = fine_label_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = (self.coarse_label_name, self.fine_label_name)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
