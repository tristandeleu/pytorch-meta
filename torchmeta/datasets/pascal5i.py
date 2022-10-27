"""
;==========================================
; Title: Pascal-5i Dataset for Few-shot Object Segmentation
; Author: Mennatullah Siam
; Company: Huawei Technologies
; Date:   18 March 2020
;==========================================
"""
import os
import json
import glob
import h5py
from PIL import Image, ImageOps

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import list_dir, download_url
from torchmeta.datasets.utils import get_asset
import numpy as np

class Pascal5i(CombinationMetaDataset):
    """
    Pascal5i dataset [1]. A dataset for few-shot object segmentation supporting 4 folds
    each fold has 15 training classes and 5 testing classes.
    Using Preprocessed Masks from [2]

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

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`,
        and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed
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

    fold : int (default: 0)
        Fold number ranges between 0-3 that controls training(15) and testing(5) classes.

    Notes
    -----
    Currently Only 1-way is supported

    References
    ----------
    .. [1] Shaban, Amirreza, et al. "One-shot learning for semantic segmentation."
            arXiv preprint arXiv:1709.03410 (2017).
    .. [2] Zhang, Chi, et al. "Canet: Class-agnostic segmentation networks with
            iterative refinement and attentive few-shot learning."
            Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_test=False, meta_split=None,
                 transform=None, target_transform=None,
                 dataset_transform=None, class_augmentations=None,
                 download=False, fold=0):

        dataset = Pascal5iClassDataset(root, meta_train=meta_train,
                                       meta_test=meta_test, transform=transform,
                                       meta_split=meta_split, class_augmentations=class_augmentations,
                                       download=download, fold=fold)

        super(Pascal5i, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)

class Pascal5iClassDataset(ClassDataset):
    folder = 'pascal5i'

    downloads = [
    {
        'url' : 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
        'filename' : 'VOCtrainval_11-May-2012.tar',
        'md5' : '6cd6e144f989b92b3379bac3b3de84fd'
    },
    {
        'url' : 'https://github.com/icoz69/CaNet/raw/master/Binary_map_aug.zip',
        'filename': 'Binary_map_aug.zip',
        'md5': None
    },
    {
        'url' : 'https://raw.github.com/NVIDIA/DIGITS/master/examples/semantic-segmentation/pascal-voc-classes.txt',
        'filename' : 'pascal-voc-classes.txt',
        'md5' : None
    }
    ]
    split_filename_labels = 'pascal-voc-classes.txt'

    def __init__(self, root, meta_train=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False, fold=0):

        super(Pascal5iClassDataset, self).__init__(meta_train=meta_train,
            meta_val=False, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.fold = fold

        self._data = None
        self._labels = None
        self._masks = None

        if download:
            self.download()

        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data, masks = self.data[0][class_name], self.data[1][class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        class_id = self.read_labels().index(class_name)

        return PascalDataset(index, (data, masks), class_id,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    def load_dict_per_class(self):
        new_exist_class_list = {}

        if self.meta_split == 'train':
            fold_list=[0, 1, 2, 3]
            fold_list.remove(self.fold)
        else:
            fold_list = [self.fold]

        for fold in fold_list:
            f = open(os.path.join(self.root, 'Binary_map_aug', self.meta_split,
                                  'split%1d_%s.txt'%(fold, self.meta_split)))
            while True:
                item = f.readline()
                if item == '':
                    break
                img_name = item[:11]
                cat = int(item[13:15])
                if cat not in new_exist_class_list:
                    new_exist_class_list[cat] = []
                new_exist_class_list[cat].append(img_name)

        images = {}
        masks = {}
        classes_names = self.read_labels()

        for k, v in new_exist_class_list.items():
            cname = classes_names[k]
            for path in v:
                fname = os.path.join(self.root, 'VOCdevkit/VOC2012/JPEGImages', path + '.jpg')
                if cname not in images:
                    images[cname] = []
                images[cname].append(fname)
                fname = os.path.join(self.root, 'Binary_map_aug', self.meta_split, str(k),
                                               path + '.png')
                if cname not in masks:
                    masks[cname] = []
                masks[cname].append(fname)
        return images, masks

    @property
    def data(self):
        if self._data is None:
            self._data, self._masks = self.load_dict_per_class()
        return self._data, self._masks

    def read_labels(self, fold=None):
        labels = []
        if fold is not None:
            if self.meta_train:
                in_classes = set(range(21)) - \
                                set(range(fold*5+1, (fold+1)*5+1))
            else:
                in_classes = set(range(fold*5+1, (fold+1)*5+1))
        else:
            in_classes = set(range(21))

        with open(os.path.join(self.root, self.split_filename_labels), 'r') as f:
            for it, line in enumerate(f):
                if line.strip() == '':
                    break
                if it in in_classes:
                    labels.append(line.strip())
        return labels

    @property
    def labels(self):
        if self._labels is None:
           self._labels = self.read_labels(self.fold)
        return self._labels[1:]

    def download(self):
        import zipfile
        import tarfile
        import shutil

        for dload in self.downloads:
            filename = os.path.join(self.root, dload['filename'])
            if os.path.isfile(filename):
                continue

            download_url(dload['url'], self.root, dload['filename'], dload['md5'])

            if 'zip' in dload['filename']:
                with zipfile.ZipFile(filename, 'r') as f:
                    f.extractall(self.root)
            elif 'tar' in dload['filename']:
                with tarfile.open(filename, 'r') as f:
                    def is_within_directory(directory, target):
                        
                        abs_directory = os.path.abspath(directory)
                        abs_target = os.path.abspath(target)
                    
                        prefix = os.path.commonprefix([abs_directory, abs_target])
                        
                        return prefix == abs_directory
                    
                    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    
                        for member in tar.getmembers():
                            member_path = os.path.join(path, member.name)
                            if not is_within_directory(path, member_path):
                                raise Exception("Attempted Path Traversal in Tar File")
                    
                        tar.extractall(path, members, numeric_owner=numeric_owner) 
                        
                    
                    safe_extract(f, self.root)

class PascalDataset(Dataset):
    def __init__(self, index, data, class_id,
                 transform=None, target_transform=None):
        super(PascalDataset, self).__init__(index, transform=transform,
            target_transform=target_transform)
        self.data, self.masks = data
        self.class_id = class_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index])
        mask = Image.open(self.masks[index])
        target = self.class_id

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return (image, mask, target)
