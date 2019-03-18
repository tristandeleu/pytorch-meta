import os
import pickle
from PIL import Image
import numpy as np

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
    gdrive_id = '1hqVbS2nhHXa51R9_aB6QDXeC0P2LQG_u'
    tar_filename = 'tiered-imagenet.tar'
    tar_md5 = '3e71dfb6cf5acecb60a29d782257da3b'
    filename_images = '{0}_images_png.pkl'
    filename_labels = '{0}_labels.pkl'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, target_transform=None,
                 class_augmentations=None, download=False):
        super(TieredImagenetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform
        self.target_transform = target_transform
        self.images_filename = os.path.join(self.root,
            self.filename_images.format(self.meta_split))
        self.labels_filename = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError()

        with open(self.images_filename, 'rb') as f:
            self._images = pickle.load(f, encoding='bytes')

        with open(self.labels_filename, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self._labels_general = data['label_general']
            self._labels_specific = data['label_specific']
            self._labels_general_str = data['label_general_str']
            self._labels_specific_str = data['label_specific_str']
        self._num_classes = np.unique(self._labels_specific).size

    def __getitem__(self, index):
        indices, = np.where(self._labels_specific == index)
        specific_class_name = self._labels_specific_str[index]
        data = [self._images[idx] for idx in indices]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index, self.target_transform)

        return TieredImagenetDataset(data, specific_class_name,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    def _check_integrity(self):
        return (os.path.isfile(os.path.join(self.root, 'class_names.txt'))
            and os.path.isfile(os.path.join(self.root, 'synsets.txt'))
            and os.path.isfile(self.images_filename)
            and os.path.isfile(self.labels_filename))

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        filename = os.path.join(self.root, self.tar_filename)
        if not os.path.isfile(filename):
            download_google_drive(self.gdrive_id, self.root, self.tar_filename)

        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)


class TieredImagenetDataset(Dataset):
    def __init__(self, data, specific_class_name, transform=None, target_transform=None):
        super(TieredImagenetDataset, self).__init__()
        self.data = data
        self.specific_class_name = specific_class_name
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        import cv2
        array = cv2.imdecode(self.data[index], 1)
        image = Image.fromarray(array)
        target = self.specific_class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
