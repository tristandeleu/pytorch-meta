import numpy as np
from PIL import Image
import os
import io
import json
import glob
import h5py

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_file_from_google_drive
from torchmeta.datasets.utils import get_asset


class TripleMNIST(CombinationMetaDataset):
    """
    The Triple MNIST dataset, introduced in [1]. This dataset is based on
    the MNIST dataset [2]. It consists of sampled images from MNIST
    that are put together to create images with multiple digits. It contains
    1,000,000 images from 1000 different classes (1000 images per class, for 
    the numbers 000 to 999).

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `triplemnist` exists.

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
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These
        classes are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the
        root directory (under the `triplemnist` folder). If the dataset is
        already available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from the Multi-digit MNIST repository
    [1](https://github.com/shaohua0116/MultiDigitMNIST). The dataset contains
    images (MNIST triple digits) from 1000 classes, for the numbers 000 to 999.
    The meta train/validation/test splits are 640/160/200 classes.
    The splits are taken from [1].

    References
    ----------
    .. [1] Sun, S. (2019). Multi-digit MNIST for Few-shot Learning.
    (https://github.com/shaohua0116/MultiDigitMNIST)

    .. [2] LeCun, Y., Cortes, C., and Burges, CJ. (2010). MNIST Handwritten
    Digit Database. (http://yann.lecun.com/exdb/mnist)

    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = TripleMNISTClassDataset(root,
            meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download)
        super(TripleMNIST, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform,
            dataset_transform=dataset_transform)


class TripleMNISTClassDataset(ClassDataset):
    folder = 'triplemnist'
    # Google Drive ID from https://github.com/shaohua0116/MultiDigitMNIST
    gdrive_id = '1xqyW289seXYaDSqD2jaBPMKVAAjPP9ee'
    zip_filename = 'triple_mnist_seed_123_image_size_84_84.zip'
    zip_md5 = '9508b047f9fbb834c02bc13ef44245da'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    image_folder = 'triple_mnist_seed_123_image_size_84_84'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(TripleMNISTClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Triple MNIST integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return TripleMNISTDataset(index, data, label, transform=transform,
                                  target_transform=target_transform)

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
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import zipfile
        import shutil
        import glob
        from tqdm import tqdm

        if self._check_integrity():
            return

        zip_filename = os.path.join(self.root, self.zip_filename)
        if not os.path.isfile(zip_filename):
            download_file_from_google_drive(self.gdrive_id, self.root,
                self.zip_filename, md5=self.zip_md5)

        zip_foldername = os.path.join(self.root, self.image_folder)
        if not os.path.isdir(zip_foldername):
            with zipfile.ZipFile(zip_filename, 'r') as f:
                for member in tqdm(f.infolist(), desc='Extracting '):
                    try:
                        f.extract(member, self.root)
                    except zipfile.BadZipFile:
                        print('Error: Zip file is corrupted')

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            labels = get_asset(self.folder, '{0}.json'.format(split))
            labels_filename = os.path.join(self.root,
                                           self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            image_folder = os.path.join(zip_foldername, split)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = glob.glob(os.path.join(image_folder, label,
                                                    '*.png'))
                    images.sort()
                    dataset = group.create_dataset(label, (len(images),),
                                                   dtype=dtype)
                    for i, image in enumerate(images):
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)

        if os.path.isdir(zip_foldername):
            shutil.rmtree(zip_foldername)


class TripleMNISTDataset(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(TripleMNISTDataset, self).__init__(index, transform=transform,
                                                 target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)
