import os
import pickle
from PIL import Image

from torch.utils.data import Dataset
from torchmeta.dataset import ClassDataset, CombinationMetaDataset
from torchmeta.datasets.utils import download_google_drive

class MiniImagenet(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MiniImagenetClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, target_transform=target_transform,
            class_augmentations=class_augmentations, download=download)
        super(MiniImagenet, self).__init__(dataset, num_classes_per_task,
            dataset_transform=dataset_transform)


class MiniImagenetClassDataset(ClassDataset):
    folder = 'miniimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    filename = 'mini-imagenet-cache-{0}.pkl'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, target_transform=None,
                 class_augmentations=None, download=False):
        super(MiniImagenetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform
        self.target_transform = target_transform
        self.pkl_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError()

        with open(self.pkl_filename, 'rb') as f:
            data = pickle.load(f)
            self._images = data['image_data']
            self._classes = data['class_dict']
            self._class_names = sorted(self._classes.keys())
        self._num_classes = len(self._classes)

    def __getitem__(self, index):
        class_name = self._class_names[index]
        indices = self._classes[class_name]
        data = self._images[indices]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index, self.target_transform)

        return MiniImagenetDataset(data, class_name, transform=transform,
            target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    def _check_integrity(self):
        return os.path.isfile(self.pkl_filename)

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        if not download_google_drive(self.gdrive_id, self.root,
                self.gz_filename, md5=self.gz_md5):
            raise RuntimeError('')

        filename = os.path.join(self.root, self.gz_filename)
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)


class MiniImagenetDataset(Dataset):
    def __init__(self, data, class_name, transform=None, target_transform=None):
        super(MiniImagenetDataset, self).__init__()
        self.data = data
        self.class_name = class_name
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
