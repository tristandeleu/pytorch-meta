import os
import pickle
from PIL import Image

from torchmeta.dataset import ClassDataset, CombinationMetaDataset
from torchmeta.datasets.miniimagenet import ImagenetDataset
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

    def __getitem__(self, index):
        class_name = None
        data = None
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index, self.target_transform)

        return ImagenetDataset(data, class_name, transform=transform,
            target_transform=target_transform)

    @property
    def num_classes(self):
        raise NotImplementedError()

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

if __name__ == '__main__':
    dataset = TieredImagenet('data', num_classes_per_task=5,
        meta_val=True, download=True)
