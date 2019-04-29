import numpy as np
from PIL import Image
import h5py
import json
import os
import io
import pickle

from torch.utils.data import Dataset
from torchmeta.dataset import ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import download_google_drive

class TieredImagenet(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=True):
        dataset = TieredImagenetClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, target_transform=target_transform,
            class_augmentations=class_augmentations, download=download)
        super(TieredImagenet, self).__init__(dataset, num_classes_per_task,
            dataset_transform=dataset_transform)


class TieredImagenetClassDataset(ClassDataset):
    folder = 'tieredimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07'
    tar_filename = 'tiered-imagenet.tar'
    tar_md5 = 'e07e811b9f29362d159a9edd0d838c62'
    tar_folder = 'tiered-imagenet'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, target_transform=None,
                 class_augmentations=None, download=False):
        super(TieredImagenetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform
        self.target_transform = target_transform

        self._data_file = None
        self._data = None
        self._labels_specific = None

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError()
        self._num_classes = len(self.labels_specific)

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels_specific(self):
        if self._labels_specific is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels_specific = json.load(f)
        return self._labels_specific

    def __getitem__(self, index):
        specific_class_name = self.labels_specific[index % self.num_classes]
        data = self.data[specific_class_name]
        general_class_name = data.attrs['label_general']
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index, self.target_transform)

        return TieredImagenetDataset(data, general_class_name, specific_class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def download(self):
        import tarfile
        import shutil
        from tqdm import tqdm

        if self._check_integrity():
            return

        if not download_google_drive(self.gdrive_id, self.root,
                self.tar_filename, md5=self.tar_md5):
            raise RuntimeError('')

        filename = os.path.join(self.root, self.tar_filename)
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)
        tar_folder = os.path.join(self.root, self.tar_folder)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            images_filename = os.path.join(tar_folder, '{0}_images_png.pkl'.format(split))
            if not os.path.isfile(images_filename):
                raise IOError(images_filename)
            with open(images_filename, 'rb') as f:
                images = pickle.load(f, encoding='bytes')

            labels_filename = os.path.join(tar_folder, '{0}_labels.pkl'.format(split))
            if not os.path.isfile(labels_filename):
                raise IOError()
            with open(labels_filename, 'rb') as f:
                labels = pickle.load(f, encoding='latin1')

            labels_str = labels['label_specific_str']
            general_labels_str = labels['label_general_str']
            general_labels = labels['label_general']
            with open(os.path.join(self.root, self.filename_labels.format(split)), 'w') as f:
                json.dump(labels_str, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels_str, desc=filename)):
                    indices, = np.where(labels['label_specific'] == i)
                    dataset = group.create_dataset(label, (len(indices),), dtype=dtype)
                    general_idx = general_labels[indices[0]]
                    dataset.attrs['label_general'] = (general_labels_str[general_idx]
                        if general_idx < len(general_labels_str) else '')
                    dataset.attrs['label_specific'] = label
                    for j, k in enumerate(indices):
                        dataset[j] = np.squeeze(images[k])

        if os.path.isdir(tar_folder):
            shutil.rmtree(tar_folder)


class TieredImagenetDataset(Dataset):
    def __init__(self, data, general_class_name, specific_class_name,
                 transform=None, target_transform=None):
        super(TieredImagenetDataset, self).__init__()
        self.data = data
        self.general_class_name = general_class_name
        self.specific_class_name = specific_class_name
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index]))
        target = (self.general_class_name, self.specific_class_name)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
