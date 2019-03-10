import os
import json
import h5py
import warnings

from torchmeta.dataset import MetaDataset
from torchmeta.tasks import Task
from torchmeta.datasets.utils import get_asset

def classproperty(msg=''):
    class _classproperty(property):
        """Subclass property to make classmethod properties possible"""
        def __get__(self, cls, owner):
            if cls is None:
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return self.fget.__get__(None, owner)()
    return _classproperty

def get_cancers():
    return get_asset(TCGA.folder, 'cancers.json', dtype='json')

def get_task_variables():
    return get_asset(TCGA.folder, 'task_variables.json', dtype='json')

class TCGA(MetaDataset):
    folder = 'tcga'
    clinical_matrix_url = 'https://tcga.xenahubs.net/download/TCGA.{0}.sampleMap/{0}_clinicalMatrix.gz'
    clinical_matrix_filename, _ = os.path.splitext(os.path.basename(clinical_matrix_url))
    gene_expression_filename = 'TCGA_HiSeqV2.hdf5'
    gene_expression_torrent = 'e4081b995625f9fc599ad860138acf7b6eb1cf6f'
    
    _task_variables = None
    _cancers = None

    def __init__(self, root, meta_train=True, min_samples_per_class=3,
                 transform=None, target_transform=None, dataset_transform=None,
                 download=False, chunksize=100, preload=True):
        super(TCGA, self).__init__(dataset_transform=dataset_transform)
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.meta_train = meta_train
        self.min_samples_per_class = min_samples_per_class

        self.transform = transform
        self.target_transform = target_transform
        
        self._all_sample_ids = None
        self._gene_ids = None

        if download:
            self.download(chunksize)

        self.preloaded = False
        self.gene_expression_file = None
        if preload:
            self._preload_gene_expression_data()
            self.preloaded = True

        self.task_ids = self.get_task_ids()

    def __len__(self):
        return len(self.task_ids)

    @property
    def gene_expression_path(self):
        filename = os.path.join(self.root, self.gene_expression_filename)
        if not os.path.isfile(filename):
            raise IOError()
        return filename
    
    @classproperty('The property `assets_path` is deprecated, and will be '
        'removed. To get the path to the assets, please use `torchmeta.'
        'datasets.utils.get_asset_path("tcga")`.')
    @classmethod
    def assets_path(cls):
        return os.path.join(os.path.dirname(__file__), 'assets', cls.folder)

    @classproperty('Access to the property `cancers` with `TCGA.cancers` is '
        'deprecated, and will be removed. To get a list of cancers from the '
        'assets, please use `tuple(torchmeta.datasets.tcga.get_cancers())`.')
    @classmethod
    def cancers(cls):
        if cls._cancers is None:
            cls._cancers = get_cancers()
        return tuple(cls._cancers)

    @classproperty('Access to the property `task_variables` with '
        '`TCGA.task_variables` is deprecated, and will be removed. To get a '
        'list of task variables from the assets, please use `tuple(torchmeta.'
        'datasets.tcga.get_task_variables())`.')
    @classmethod
    def task_variables(cls):
        if cls._task_variables is None:
            cls._task_variables = frozenset(get_task_variables())
        return tuple(cls._task_variables)

    @property
    def gene_ids(self):
        if self._gene_ids is None:
            gene_ids_file = os.path.join(self.root, 'gene_ids.json')
            if not os.path.isfile(gene_ids_file):
                raise IOError()
            with open(gene_ids_file, 'r') as f:
                self._gene_ids = set(json.load(f))
        return self._gene_ids

    @property
    def all_sample_ids(self):
        if self._all_sample_ids is None:
            all_sample_ids_file = os.path.join(self.root, 'all_sample_ids.json')
            if not os.path.isfile(all_sample_ids_file):
                raise IOError()
            with open(all_sample_ids_file, 'r') as f:
                all_sample_ids = json.load(f)
            self._all_sample_ids = dict((v, k) for (k, v) in enumerate(all_sample_ids))
        return self._all_sample_ids

    def get_processed_filename(self, cancer):
        processed_folder = os.path.join(self.root, 'clinicalMatrices', 'processed')
        filename = '{0}.tsv'.format(self.clinical_matrix_filename.format(cancer))
        filepath = os.path.join(processed_folder, filename)
        if not os.path.isfile(filepath):
            raise IOError()
        return filepath

    def __getitem__(self, index):
        import pandas as pd
        label, cancer = self.task_ids[index]
        filename = self.get_processed_filename(cancer)
        dataframe = pd.read_csv(filename, sep='\t', index_col=0, header=0)
        labels = dataframe[label].dropna().astype('category')

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

    def get_task_ids(self):
        import pandas as pd
        clinical_matrices_folder = os.path.join(self.root, 'clinicalMatrices')
        processed_folder = os.path.join(clinical_matrices_folder, 'processed')
        if not os.path.exists(processed_folder):
            os.makedirs(processed_folder)

        col_in_task_variables = lambda col: (col == 'sampleID') or (col in self.task_variables)

        task_ids = []
        for cancer in self.cancers:
            filename = self.clinical_matrix_filename.format(cancer)
            filepath = os.path.join(clinical_matrices_folder, '{0}.tsv'.format(filename))
            processed = os.path.join(processed_folder, '{0}.tsv'.format(filename))

            if not os.path.isfile(processed):
                raw_df = pd.read_csv(filepath, sep='\t', index_col=0, header=0,
                    usecols=col_in_task_variables)#.dropna(axis=0, how='any')
                dataframe = raw_df[raw_df.index.isin(self.all_sample_ids)]
                dataframe.index = dataframe.index.map(lambda index: self.all_sample_ids[index])
                dataframe.index.names = ['index']
                dataframe = dataframe.sort_index(axis=0)
                dataframe.to_csv(processed, sep='\t')
            else:
                dataframe = pd.read_csv(processed, sep='\t', index_col=0, header=0)

            num_samples_per_label = dataframe.apply(pd.value_counts)
            min_samples_per_class = num_samples_per_label.min(axis=0)
            count_classes = num_samples_per_label.count(axis=0)
            labels = min_samples_per_class[(min_samples_per_class > self.min_samples_per_class) & (count_classes > 1)]

            task_ids.extend([(label, cancer) for label in labels.index])

        return task_ids

    def download(self, chunksize=100):
        import gzip
        import shutil
        import pandas as pd
        from six.moves import urllib

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
            import academictorrents as at
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

        # Clean up
        for cancer in self.cancers:
            filename = self.clinical_matrix_filename.format(cancer)
            rawpath = os.path.join(clinical_matrices_folder, '{0}.gz'.format(filename))
            if os.path.isfile(rawpath):
                os.remove(rawpath)

    def close(self):
        if self.preloaded:
            self.gene_expression_file.close()
            self.gene_expression_file = None
            self.preloaded = False


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
    
    def __init__(self, task_id, data, labels, categories, transform=None,
                 target_transform=None):
        super(TCGATask, self).__init__(len(categories))
        self.id = task_id
        self.data = data
        self.labels = labels
        self.categories = categories

        self.transform = transform
        self.target_transform = target_transform

    @property
    def input_size(self):
        return len(self.data[0])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target)
