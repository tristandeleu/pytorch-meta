## Omniglot

The Omniglot dataset [1]. A dataset of 1623 handwritten characters from 50 different alphabets.

```python
torchmeta.datasets.Omniglot(root, num_classes_per_task=None, meta_train=False,
    meta_val=False, meta_test=False, meta_split=None, use_vinyals_split=True,
    transform=None, target_transform=None, dataset_transform=None,
    class_augmentations=None, download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `omniglot` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to "N" in "N-way" classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **use_vinyals_split**: *bool (default: `True`)*
 If set to `True`, the dataset uses the splits defined in [3]. If `False`, then the meta-train split corresponds to `images_background`, and the meta-test split corresponds to `images_evaluation` (raises an error when calling the meta-validation split).

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `torchmeta.transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the zip files and processes the dataset in the root directory (under the `omniglot` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The dataset is downloaded from the original [Omniglot repository](https://github.com/brendenlake/omniglot). The meta train/validation/test splits used in [3] are taken from [this repository](https://github.com/jakesnell/prototypical-networks). These splits are over 1028/172/423 classes (characters).

!!! attention "References"
    - **[1]** Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2015). Human-level concept learning through probabilistic program induction. Science, 350(6266), 1332-1338 (http://www.sciencemag.org/content/350/6266/1332.short)
    - **[2]** Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. (2019). The Omniglot Challenge: A 3-Year Progress Report (https://arxiv.org/abs/1902.03477)
    - **[3]** Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016). Matching Networks for One Shot Learning. In Advances in Neural Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)

## MiniImagenet

The Mini-Imagenet dataset, introduced in [1]. This dataset contains images of 100 different classes from the ILSVRC-12 dataset (Imagenet challenge). The meta train/validation/test splits are taken from [2] for reproducibility.

```python
torchmeta.datasets.MiniImagenet(root, num_classes_per_task=None,
    meta_train=False, meta_val=False, meta_test=False, meta_split=None,
    transform=None, target_transform=None, dataset_transform=None,
    class_augmentations=None, download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `miniimagenet` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to "N" in "N-way" classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `torchmeta.transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the pickle files and processes the dataset in the root directory (under the `miniimagenet` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The dataset is downloaded from [this repository](https://github.com/renmengye/few-shot-ssl-public/). The meta train/validation/test splits are over 64/16/20 classes.

!!! attention "References"
    - **[1]** Vinyals, O., Blundell, C., Lillicrap, T. and Wierstra, D. (2016). Matching Networks for One Shot Learning. In Advances in Neural Information Processing Systems (pp. 3630-3638) (https://arxiv.org/abs/1606.04080)
    - **[2]** Ravi, S. and Larochelle, H. (2016). Optimization as a Model for Few-Shot Learning. (https://openreview.net/forum?id=rJY0-Kcll)

## TieredImagenet

The Tiered-Imagenet dataset, introduced in [1]. This dataset contains images of 608 different classes from the ILSVRC-12 dataset (Imagenet challenge).

```python
torchmeta.datasets.TieredImagenet(root, num_classes_per_task=None,
    meta_train=False, meta_val=False, meta_test=False, meta_split=None,
    transform=None, target_transform=None, dataset_transform=None,
    class_augmentations=None, download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `tieredimagenet` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to "N" in "N-way" classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `torchmeta.transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the pickle files and processes the dataset in the root directory (under the `tieredimagenet` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The dataset is downloaded from [this repository](https://github.com/renmengye/few-shot-ssl-public/). The dataset contains images from 34 categories. The meta train/validation/test splits are over 20/6/8 categories. Each category contains between 10 and 30 classes. The splits over categories (instead of over classes) ensures that all the training classes are sufficiently distinct from the test classes (unlike Mini-Imagenet).

!!! attention "References"
    - **[1]** Ren, M., Triantafillou, E., Ravi, S., Snell, J., Swersky, K., Tenenbaum, J.B., Larochelle, H. and Zemel, R.S. (2018). Meta-learning for semi-supervised few-shot classification. International Conference on Learning Representations. (https://arxiv.org/abs/1803.00676)

## FC100

The Fewshot-CIFAR100 dataset, introduced in [1]. This dataset contains images of 100 different classes from the CIFAR100 dataset [2].

```python
torchmeta.datasets.FC100(root, num_classes_per_task=None, meta_train=False,
    meta_val=False, meta_test=False, meta_split=None, transform=None,
    target_transform=None, dataset_transform=None, class_augmentations=None,
    download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `cifar100` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to `N` in `N-way` classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the pickle files and processes the dataset in the root directory (under the `cifar100` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The meta train/validation/test splits are over 12/4/4 superclasses from the CIFAR100 dataset. The meta train/validation/test splits contain 60/20/20 classes.

!!! attention "References"
    - **[1]** Oreshkin B. N., Rodriguez P., Lacoste A. (2018). TADAM: Task dependent adaptive metric for improved few-shot learning. In Advances in Neural Information Processing Systems (https://arxiv.org/abs/1805.10123)
    - **[2]** Krizhevsky A. (2009). Learning Multiple Layers of Features from Tiny Images. (https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## CIFARFS

The CIFAR-FS dataset, introduced in [1]. This dataset contains images of 100 different classes from the CIFAR100 dataset [2].

```python
torchmeta.datasets.CIFARFS(root, num_classes_per_task=None, meta_train=False,
    meta_val=False, meta_test=False, meta_split=None, transform=None,
    target_transform=None, dataset_transform=None, class_augmentations=None,
    download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `cifar100` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to `N` in `N-way` classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the pickle files and processes the dataset in the root directory (under the `cifar100` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The meta train/validation/test splits are over 64/16/20 classes from the CIFAR100 dataset.

!!! attention "References"
    - **[1]** Bertinetto L., Henriques J. F., Torr P. H.S., Vedaldi A. (2019). Meta-learning with differentiable closed-form solvers. In International Conference on Learning Representations (https://arxiv.org/abs/1805.08136)
    - **[2]** Krizhevsky A. (2009). Learning Multiple Layers of Features from Tiny Images. (https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## CUB

The Caltech-UCSD Birds dataset, introduced in [1]. This dataset is based on images from 200 species of birds from the Caltech-UCSD Birds dataset [2].

```python
torchmeta.datasets.CUB(root, num_classes_per_task=None, meta_train=False,
    meta_val=False, meta_test=False, meta_split=None, transform=None,
    target_transform=None, dataset_transform=None, class_augmentations=None,
    download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `cub` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to "N" in "N-way" classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `torchmeta.transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the pickle files and processes the dataset in the root directory (under the `cub` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The dataset is downloaded from [2]. The dataset contains images from 200 classes. The meta train/validation/test splits are over 100/50/50 classes. The splits are taken from [3] ([code](https://github.com/wyharveychen/CloserLookFewShot)for reproducibility).

!!! attention "References"
    - **[1]** Hilliard, N., Phillips, L., Howland, S., Yankov, A., Corley, C. D., Hodas, N. O. (2018). Few-Shot Learning with Metric-Agnostic Conditional Embeddings. (https://arxiv.org/abs/1802.04376)
    - **[2]** Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S. (2011). The Caltech-UCSD Birds-200-2011 Dataset (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    - **[3]** Chen, W., Liu, Y. and Kira, Z. and Wang, Y. and  Huang, J. (2019). A Closer Look at Few-shot Classification. International Conference on Learning Representations (https://openreview.net/forum?id=HkxLXnAcFQ)

## DoubleMNIST

The Double MNIST dataset, introduced in [1]. This dataset is based on the MNIST dataset [2]. It consists of sampled images from MNIST that are put together to create images with multiple digits. It contains 100,000 images from 100 different classes (1000 images per class, for the numbers 00 to 99).

```python
torchmeta.datasets.DoubleMNIST(root, num_classes_per_task=None,
    meta_train=False, meta_val=False, meta_test=False, meta_split=None,
    transform=None, target_transform=None, dataset_transform=None,
    class_augmentations=None, download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `doublemnist` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to "N" in "N-way" classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `torchmeta.transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the pickle files and processes the dataset in the root directory (under the `doublemnist` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The dataset is downloaded from the Multi-digit MNIST repository [1](https://github.com/shaohua0116/MultiDigitMNIST). The dataset contains images (MNIST double digits) from 100 classes, for the numbers 00 to 99. The meta train/validation/test splits are 64/16/20 classes. The splits are taken from [1].

!!! attention "References"
    - **[1]** Sun, S. (2019). Multi-digit MNIST for Few-shot Learning. (https://github.com/shaohua0116/MultiDigitMNIST)
    - **[2]** LeCun, Y., Cortes, C., and Burges, CJ. (2010). MNIST Handwritten Digit Database. (http://yann.lecun.com/exdb/mnist)

## TripleMNIST

The Triple MNIST dataset, introduced in [1]. This dataset is based on the MNIST dataset [2]. It consists of sampled images from MNIST that are put together to create images with multiple digits. It contains 1,000,000 images from 1000 different classes (1000 images per class, for the numbers 000 to 999).

```python
torchmeta.datasets.TripleMNIST(root, num_classes_per_task=None,
    meta_train=False, meta_val=False, meta_test=False, meta_split=None,
    transform=None, target_transform=None, dataset_transform=None,
    class_augmentations=None, download=False)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `triplemnist` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to "N" in "N-way" classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `torchmeta.transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the pickle files and processes the dataset in the root directory (under the `triplemnist` folder). If the dataset is already available, this does not download/process the dataset again.

!!! note "Notes"
    The dataset is downloaded from the Multi-digit MNIST repository [1](https://github.com/shaohua0116/MultiDigitMNIST). The dataset contains images (MNIST triple digits) from 1000 classes, for the numbers 000 to 999. The meta train/validation/test splits are 640/160/200 classes. The splits are taken from [1].

!!! attention "References"
    - **[1]** Sun, S. (2019). Multi-digit MNIST for Few-shot Learning. (https://github.com/shaohua0116/MultiDigitMNIST)
    - **[2]** LeCun, Y., Cortes, C., and Burges, CJ. (2010). MNIST Handwritten Digit Database. (http://yann.lecun.com/exdb/mnist)

## TCGA

The TCGA dataset [1]. A dataset of classification tasks over the values of an attribute, based on the gene expression data from patients diagnosed with specific types of cancer. This dataset is based on data from the Cancer Genome Atlas Program from the National Cancer Institute.

```python
torchmeta.datasets.TCGA(root, meta_train=False, meta_val=False,
    meta_test=False, meta_split=None, min_samples_per_class=5, transform=None,
    target_transform=None, dataset_transform=None, download=False,
    chunksize=100, preload=True)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `omniglot` exists.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_val**: *bool (default: `False`)*
 Use the meta-validation split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'val', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, `meta_val` and `meta_test` if all three are set to `False`.

 - **min_samples_per_class**: *int (default: 5)*
 Minimum number of samples per class in each classification task. This filters tasks for which the amount of data for one of the classes is too small.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **target_transform**: *callable, optional*
 A function/transform that takes a target, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `transforms.ClassSplitter()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the files and processes the dataset in the root directory (under the `tcga` folder). If the dataset is already available, this does not download/process the dataset again.

 - **chunksize**: *int (default: 100)*
 Size of the chunks to be processed when reading the CSV file. This is only used while downloading and converting the dataset to HDF5.

 - **preload**: *bool (default: `True`)*
 Opens the gene expression dataset and keeps a reference to it in memory. This decreases the loading time of individual tasks.

!!! note "Notes"
    A task is the combination of a cancer type and an attribute. The data is the gene expression of patients diagnosed with the cancer defined by the task. It consists in a vector of size `(20530,)`. The task is to classify the patients according to the attribute given by the task definition. The meta train/validation/test splits are over 137/29/29 tasks (ie. types of cancer). However, the number of tasks depends on the minimum number of samples per class specified by `min_samples_per_class`.

!!! attention "References"
    - **[1]** Samiei, M., Wurfl, T., Deleu, T., Weiss, M., Dutil, F., Fevens, T., Boucher, G., Lemieux, S., and Cohen, J. P. (2019). The TCGA Meta-Dataset Clinical Benchmark. (https://arxiv.org/abs/1910.08636)

## Pascal5i

Pascal5i dataset [1]. A dataset for few-shot object segmentation supporting 4 folds each fold has 15 training classes and 5 testing classes. Using Preprocessed Masks from [2]

```python
torchmeta.datasets.Pascal5i(root, num_classes_per_task=None, meta_train=False,
    meta_test=False, meta_split=None, transform=None, target_transform=None,
    dataset_transform=None, class_augmentations=None, download=False, fold=0)
```

**Parameters**

 - **root**: *string*
 Root directory where the dataset folder `omniglot` exists.

 - **num_classes_per_task**: *int*
 Number of classes per tasks. This corresponds to "N" in "N-way" classification.

 - **meta_train**: *bool (default: `False`)*
 Use the meta-train split of the dataset. If set to `True`, then the arguments `meta_val` and `meta_test` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_test**: *bool (default: `False`)*
 Use the meta-test split of the dataset. If set to `True`, then the arguments `meta_train` and `meta_val` must be set to `False`. Exactly one of these three arguments must be set to `True`.

 - **meta_split**: *string in {'train', 'test'}, optional*
 Name of the split to use. This overrides the arguments `meta_train`, and `meta_test` if all three are set to `False`.

 - **transform**: *callable, optional*
 A function/transform that takes a `PIL` image, and returns a transformed version. See also `torchvision.transforms`.

 - **dataset_transform**: *callable, optional*
 A function/transform that takes a dataset (ie. a task), and returns a transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

 - **class_augmentations**: *list of callable, optional*
 A list of functions that augment the dataset with new classes. These classes are transformations of existing classes. E.g. `torchmeta.transforms.HorizontalFlip()`.

 - **download**: *bool (default: `False`)*
 If `True`, downloads the zip files and processes the dataset in the root directory (under the `omniglot` folder). If the dataset is already available, this does not download/process the dataset again.

 - **fold**: *int (default: 0)*
 Fold number ranges between 0-3 that controls training(15) and testing(5) classes.

!!! note "Notes"
    Currently Only 1-way is supported

!!! attention "References"
    - **[1]** Shaban, Amirreza, et al. "One-shot learning for semantic segmentation." arXiv preprint arXiv:1709.03410 (2017).
    - **[2]** Zhang, Chi, et al. "Canet: Class-agnostic segmentation networks with iterative refinement and attentive few-shot learning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.