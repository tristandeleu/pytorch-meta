import os
import json
import h5py
import numpy as np
import torch
import copy

from torchmeta.utils.data import Task, MetaDataset
from torchmeta.datasets.utils import get_asset


class TCGA(MetaDataset):
    """
    The TCGA dataset [1]. A dataset of classification tasks over the values of
    an attribute, based on the gene expression data from patients diagnosed with
    specific types of cancer. This dataset is based on data from the Cancer
    Genome Atlas Program from the National Cancer Institute.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `omniglot` exists.

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

    min_samples_per_class : int (default: 5)
        Minimum number of samples per class in each classification task. This
        filters tasks for which the amount of data for one of the classes is
        too small.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `transforms.ClassSplitter()`.

    download : bool (default: `False`)
        If `True`, downloads the files and processes the dataset in the root 
        directory (under the `tcga` folder). If the dataset is already 
        available, this does not download/process the dataset again.

    chunksize : int (default: 100)
        Size of the chunks to be processed when reading the CSV file. This is
        only used while downloading and converting the dataset to HDF5.

    preload : bool (default: `True`)
        Opens the gene expression dataset and keeps a reference to it in memory.
        This decreases the loading time of individual tasks.

    Notes
    -----
    A task is the combination of a cancer type and an attribute. The data is the
    gene expression of patients diagnosed with the cancer defined by the task.
    It consists in a vector of size `(20530,)`. The task is to classify the
    patients according to the attribute given by the task definition. The meta
    train/validation/test splits are over 137/29/29 tasks (ie. types of cancer).
    However, the number of tasks depends on the minimum number of samples per
    class specified by `min_samples_per_class`.

    References
    ----------
    .. [1] Samiei, M., Wurfl, T., Deleu, T., Weiss, M., Dutil, F., Fevens, T.,
           Boucher, G., Lemieux, S., and Cohen, J. P. (2019). The TCGA
           Meta-Dataset Clinical Benchmark. (https://arxiv.org/abs/1910.08636)
    """
    folder = 'tcga'
    clinical_matrix_url = 'https://tcga.xenahubs.net/download/TCGA.{0}.sampleMap/{0}_clinicalMatrix.gz'
    clinical_matrix_filename, _ = os.path.splitext(os.path.basename(clinical_matrix_url))
    gene_expression_filename = 'TCGA_HiSeqV2.hdf5'
    gene_expression_torrent = 'e4081b995625f9fc599ad860138acf7b6eb1cf6f'

    filename_tasks = '{0}_labels.json'

    _task_variables = None
    _cancers = None

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False, meta_split=None,
                 min_samples_per_class=5, transform=None, target_transform=None,
                 dataset_transform=None, download=False, chunksize=100, preload=True):
        super(TCGA, self).__init__(meta_train, meta_val, meta_test, meta_split,
            target_transform=target_transform, dataset_transform=dataset_transform)
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.min_samples_per_class = min_samples_per_class
        self.transform = transform

        self._all_sample_ids = None
        self._gene_ids = None
        self._tasks = None

        if download:
            self.download(chunksize)

        self.preloaded = False
        self.gene_expression_data = None
        self.gene_expression_file = None
        if preload:
            self._preload_gene_expression_data()
            self.preloaded = True

        self.task_ids = self.get_task_ids()
        self.split_filename_tasks = os.path.join(self.root,
            self.filename_tasks.format(self.meta_split))

    def __len__(self):
        return len(self.task_ids)

    @property
    def gene_expression_path(self):
        filename = os.path.join(self.root, self.gene_expression_filename)
        if not os.path.isfile(filename):
            raise IOError('Gene expression data not found at {}'.format(filename))
        return filename

    @property
    def tasks(self):
        if self._tasks is None:
            with open(self.split_filename_tasks, 'r') as f:
                self._tasks = [task for task in json.load(f) if tuple(task) in self.task_ids]
        return self._tasks

    @property
    def cancers(self):
        if self._cancers is None:
            self._cancers = get_cancers()
        return self._cancers

    @property
    def task_variables(self):
        if self._task_variables is None:
            self._task_variables = frozenset(get_task_variables())
        return self._task_variables

    @property
    def gene_ids(self):
        if self._gene_ids is None:
            gene_ids_file = os.path.join(self.root, 'gene_ids.json')
            if not os.path.isfile(gene_ids_file):
                raise IOError('Gene id data not found at {}'.format(gene_ids_file))
            with open(gene_ids_file, 'r') as f:
                self._gene_ids = set(json.load(f))
        return self._gene_ids

    @property
    def all_sample_ids(self):
        if self._all_sample_ids is None:
            all_sample_ids_file = os.path.join(self.root, 'all_sample_ids.json')
            if not os.path.isfile(all_sample_ids_file):
                raise IOError('All sample id data not found at {}'.format(all_sample_ids_file))
            with open(all_sample_ids_file, 'r') as f:
                all_sample_ids = json.load(f)
            self._all_sample_ids = dict((v, k) for (k, v) in enumerate(all_sample_ids))
        return self._all_sample_ids

    def get_processed_filename(self, cancer):
        processed_folder = os.path.join(self.root, 'clinicalMatrices', 'processed')
        filename = '{0}.tsv'.format(self.clinical_matrix_filename.format(cancer))
        filepath = os.path.join(processed_folder, filename)
        if not os.path.isfile(filepath):
            raise IOError('Clinical matrix file not found at {}'.format(filepath))
        return filepath

    def __getitem__(self, index):
        import pandas as pd

        label, cancer = self.tasks[index]
        filename = self.get_processed_filename(cancer)
        dataframe = pd.read_csv(filename, sep='\t', index_col=0, header=0)
        labels = dataframe[label].dropna().astype('category')
        labels = labels[self.task_ids[(label, cancer)]]

        if self.gene_expression_file is not None:
            data = self.gene_expression_data[labels.index]
        else:
            with h5py.File(self.gene_expression_path, 'r') as f:
                data = f['expression_data'][labels.index]

        task = TCGATask((label, cancer), data, labels.cat.codes.tolist(),
                        labels.cat.categories.tolist(), transform=self.transform,
                        target_transform=self.target_transform)

        if self.dataset_transform is not None:
            task = self.dataset_transform(task)

        return task

    def _preload_gene_expression_data(self):
        self.gene_expression_file = h5py.File(self.gene_expression_path, 'r')
        self.gene_expression_data = self.gene_expression_file['expression_data']

    def _process_clinical_matrices(self):
        import pandas as pd
        clinical_matrices_folder = os.path.join(self.root, 'clinicalMatrices')
        processed_folder = os.path.join(clinical_matrices_folder, 'processed')
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        col_in_task_variables = lambda col: (col == 'sampleID') or (col in self.task_variables)

        for cancer in self.cancers:
            filename = self.clinical_matrix_filename.format(cancer)
            filepath = os.path.join(clinical_matrices_folder, '{0}.tsv'.format(filename))
            processed = os.path.join(processed_folder, '{0}.tsv'.format(filename))

            if not os.path.isfile(processed):
                raw_df = pd.read_csv(filepath, sep='\t', index_col=0, header=0,
                                     usecols=col_in_task_variables)
                dataframe = raw_df[raw_df.index.isin(self.all_sample_ids)]
                dataframe.index = dataframe.index.map(lambda index: self.all_sample_ids[index])
                dataframe.index.names = ['index']
                dataframe = dataframe.sort_index(axis=0)
                dataframe.to_csv(processed, sep='\t')
        return True

    def get_task_ids(self):
        tasks = get_task_id_splits(self.meta_split)
        task_ids = dict()

        for task_id in tasks:
            indices, counts = tasks[task_id]
            enough_samples = all(count > self.min_samples_per_class for count in counts.values())
            if enough_samples:
                task_id = tuple(task_id.split('|', 1))
                task_ids[task_id] = indices

        return task_ids

    def download(self, chunksize=100):
        try:
            import gzip
            import shutil
            import pandas as pd
            from six.moves import urllib
            import academictorrents as at
        except ImportError as exception:
            raise ImportError('{0}. To use the TCGA dataset, you need to '
                'install the necessary dependencies with '
                '`pip install torchmeta[tcga]`.'.format(exception.message))

        clinical_matrices_folder = os.path.join(self.root, 'clinicalMatrices')
        if not os.path.exists(clinical_matrices_folder):
            os.makedirs(clinical_matrices_folder)

        for cancer in self.cancers:
            filename = self.clinical_matrix_filename.format(cancer)
            rawpath = os.path.join(clinical_matrices_folder, '{0}.gz'.format(filename))
            filepath = os.path.join(clinical_matrices_folder, '{0}.tsv'.format(filename))

            if os.path.isfile(filepath):
                continue

            if not os.path.exists(rawpath):
                print('Downloading `{0}.gz`...'.format(filename))
                url = self.clinical_matrix_url.format(cancer)
                urllib.request.urlretrieve(url, rawpath)

            print('Extracting `{0}.gz`...'.format(filename))
            with gzip.open(rawpath, 'rb') as gzf:
                with open(filepath, 'wb') as f:
                    shutil.copyfileobj(gzf, f)

        gene_expression_file = os.path.join(self.root, self.gene_expression_filename)
        if not os.path.isfile(gene_expression_file):
            from tqdm import tqdm
            print('Downloading `{0}` using `academictorrents`...'.format(
                self.gene_expression_filename))
            csv_file = at.get(self.gene_expression_torrent, datastore=self.root)
            print('Downloaded to: `{0}`'.format(csv_file))

            print('Converting TCGA CSV dataset to HDF5. This may take a while, '
                  'but only happens on the first run.')
            reader = pd.read_csv(csv_file, compression='gzip', sep='\t',
                                 header=0, index_col=0, chunksize=chunksize)
            shape = (10459, 20530)

            with tqdm(total=shape[1]) as pbar:
                with h5py.File(gene_expression_file, 'w') as f:
                    dataset = f.create_dataset('expression_data',
                                               shape=shape, dtype='f4')
                    gene_ids = []
                    for idx, chunk in enumerate(reader):
                        slice_ = slice(idx * chunksize, (idx + 1) * chunksize)
                        dataset[:, slice_] = chunk.T
                        gene_ids.extend(chunk.index)
                        pbar.update(chunk.shape[0])
                    all_sample_ids = chunk.columns.tolist()

            gene_ids_file = os.path.join(self.root, 'gene_ids.json')
            with open(gene_ids_file, 'w') as f:
                json.dump(gene_ids, f)

            all_sample_ids_file = os.path.join(self.root, 'all_sample_ids.json')
            with open(all_sample_ids_file, 'w') as f:
                json.dump(all_sample_ids, f)

            if os.path.isfile(csv_file):
                os.remove(csv_file)

            print('Done')

        self._process_clinical_matrices()

        # Create label files
        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename_tasks.format(split))
            data = get_asset(self.folder, '{0}.json'.format(split), dtype='json')

            with open(filename, 'w') as f:
                labels = sorted([key.split('|', 1) for key in data])
                json.dump(labels, f)

        # Clean up
        for cancer in self.cancers:
            filename = self.clinical_matrix_filename.format(cancer)
            rawpath = os.path.join(clinical_matrices_folder, '{0}.gz'.format(filename))
            if os.path.isfile(rawpath):
                os.remove(rawpath)

    def close(self):
        if self.preloaded:
            self.gene_expression_file.close()
            self.gene_expression_data = None
            self.gene_expression_file = None
            self.preloaded = False

    def open(self):
        if self.preloaded:
            self._preload_gene_expression_data()
            self.preloaded = True


class TCGATask(Task):
    @classmethod
    def from_id(cls, root, task_id, transform=None, target_transform=None):
        import pandas as pd
        root = os.path.join(os.path.expanduser(root), TCGA.folder)
        gene_filepath = os.path.join(root, TCGA.gene_expression_filename)
        if not os.path.isfile(gene_filepath):
            raise IOError()

        label, cancer = task_id

        processed_folder = os.path.join(root, 'clinicalMatrices', 'processed')
        filename = '{0}.tsv'.format(TCGA.clinical_matrix_filename.format(cancer))
        filepath = os.path.join(processed_folder, filename)
        if not os.path.isfile(filepath):
            raise IOError()

        dataframe = pd.read_csv(filepath, sep='\t', index_col=0, header=0)
        labels = dataframe[label].dropna().astype('category')

        with h5py.File(gene_filepath, 'r') as f:
            data = f['expression_data'][labels.index]

        return cls(task_id, data, labels.cat.codes.tolist(),
                   labels.cat.categories.tolist(), transform=transform,
                   target_transform=target_transform)

    def __init__(self, task_id, data, labels, categories,
                 transform=None, target_transform=None):
        super(TCGATask, self).__init__(task_id, len(categories),
            transform=transform, target_transform=target_transform)
        self.id = task_id
        self.data = data
        self.labels = labels
        self.categories = categories

    @property
    def input_size(self):
        return len(self.data[0])

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target)


def _assign_samples(tcga_metadataset):
    import pandas as pd
    import munkres

    blacklist = []
    sample_to_task_assignment = {}
    for cancer in get_cancers():
        filename = tcga_metadataset.get_processed_filename(cancer)
        dataframe = pd.read_csv(filename, sep='\t', index_col=0, header=0)
        dataframe = dataframe.drop(blacklist, errors='ignore')
        permutation = dataframe.index[torch.randperm(len(dataframe.index))]
        dataframe = dataframe.reindex(permutation)
        labels = dataframe.notna()
        labels = labels.applymap(lambda x: 1. if x else munkres.DISALLOWED)
        all_disallowed = labels.apply(lambda x: True if all(x == munkres.DISALLOWED) else False, axis=1)
        labels = labels.drop(labels[all_disallowed].index)

        matrix = labels.values
        shape = matrix.shape
        # The +5 allows for some slack in the assignment
        # which is necessary for the used implementation to converge on BRCA
        repeats = np.int(np.ceil(shape[0] / shape[1])) + 5
        expanded_matrix = np.tile(matrix, (1, repeats))

        indices = munkres.Munkres().compute(expanded_matrix)
        mapped_indices = [(a, b % shape[1]) for a, b in indices]

        for index, mapped_index in mapped_indices:
            sample_to_task_assignment.setdefault((dataframe.columns[mapped_index], cancer), []).append(
                dataframe.index[index])

        blacklist.extend(dataframe.index.tolist())

    return sample_to_task_assignment


def _expand_sample_usage(meta_dataset, all_allowed_samples, additional_samples):
    expanded_metadataset = {}
    all_samples_of_metadataset = set()
    for key, value in meta_dataset.items():
        all_samples_of_metadataset.update(value)
    all_samples_of_metadataset.update(additional_samples)

    used_additional_samples = set()
    for key in meta_dataset.keys():
        allowed_samples = set(all_allowed_samples[key])
        intersection = allowed_samples.intersection(all_samples_of_metadataset)
        expanded_metadataset[key] = list(intersection)
        used_additional_samples = additional_samples.intersection(intersection)

    return expanded_metadataset, used_additional_samples


def _split_tcga(tcga_metadataset, counts):
    all_allowed_samples = tcga_metadataset.task_ids

    # We first uniquely assing every sample to a task
    sample_to_task_assignment = _assign_samples(tcga_metadataset)

    keys = [i for i in all_allowed_samples.keys()]
    difference = set(sample_to_task_assignment.keys()).difference(set(keys))

    unassigned_samples = set()
    for key in difference:
        unassigned_samples.update(sample_to_task_assignment[key])

    # Second we split the metadataset
    # with a torch-based random sample
    permutation = torch.randperm(len(keys)).numpy()

    metadatasets = []
    start = 0
    end = 0
    for count in counts:
        end += count
        current_keys = [keys[index] for index in permutation[start:end]]
        metadatasets.append({key: sample_to_task_assignment[key] for key in current_keys})
        start = end

    expanded_metadatasets = [None] * len(metadatasets)
    order = np.argsort([len(metadataset) for metadataset in metadatasets])

    # Finally we expand the tasks by reusing samples wherever possible in the sets
    blacklist = set()
    for i in order:
        additional_samples = unassigned_samples.difference(blacklist)
        expanded_metadataset, used_additional_samples = _expand_sample_usage(metadatasets[i], all_allowed_samples,
                                                                             additional_samples)
        expanded_metadatasets[i] = (expanded_metadataset)
        blacklist.update(used_additional_samples)

    tcga_metadatasets = []
    tcga_metadataset.close()
    preloaded = tcga_metadataset.preloaded
    for metadataset in expanded_metadatasets:
        current_tcga_metadataset = copy.deepcopy(tcga_metadataset)
        current_tcga_metadataset.task_ids = metadataset
        if preloaded:
            current_tcga_metadataset.open()
        tcga_metadatasets.append(current_tcga_metadataset)

    return tcga_metadatasets


def get_cancers():
    return get_asset(TCGA.folder, 'cancers.json', dtype='json')


def get_task_variables():
    return get_asset(TCGA.folder, 'task_variables.json', dtype='json')


def get_task_id_splits(meta_split):
    return get_asset(TCGA.folder, '{}.json'.format(meta_split), dtype='json')
