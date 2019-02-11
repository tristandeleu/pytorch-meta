import os
import json
import h5py
import pandas as pd

from torchmeta.dataset import Dataset, Task

class TCGA(Dataset):
    folder = 'datasets' #'tcga'
    clinical_matrix_url = 'https://tcga.xenahubs.net/download/TCGA.{0}.sampleMap/{0}_clinicalMatrix.gz'
    clinical_matrix_filename, _ = os.path.splitext(os.path.basename(clinical_matrix_url))
    gene_expression_filename = 'TCGA_tissue_ppi.hdf5'
    gene_expression_torrent = '4070a45bc7dd69584f33e86ce193a2c903f0776d'

    def __init__(self, root, meta_train=True, min_samples_per_class=3, transform=None,
                 target_transform=None, download=False, preload=True):
        super(TCGA, self).__init__(class_transforms=None)
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.meta_train = meta_train
        self.min_samples_per_class = min_samples_per_class

        self.transform = transform
        self.target_transform = target_transform

        self.assets_path = os.path.join(os.path.dirname(__file__), 'assets', 'tcga') #self.folder)
        self._cancers = None
        self._task_variables = None
        self._all_sample_ids = None
        self._gene_ids = None

        if download:
            self.download()

        self.preloaded = False
        self.gene_expression_file = None
        if preload:
            self._preload_gene_expression_data()
            self.preloaded = True

        self.task_ids = self._get_task_ids()

    @property
    def num_classes(self):
        return len(self.task_ids)

    @property
    def gene_expression_path(self):
        filename = os.path.join(self.root, self.gene_expression_filename)
        if not os.path.isfile(filename):
            raise IOError()
        return filename

    @property
    def cancers(self):
        if self._cancers is None:
            filename = os.path.join(self.assets_path, 'cancers.json')
            if not os.path.isfile(filename):
                raise IOError()
            with open(filename, 'r') as f:
                self._cancers = json.load(f)
        return self._cancers

    @property
    def task_variables(self):
        if self._task_variables is None:
            filename = os.path.join(self.assets_path, 'task_variables.json')
            if not os.path.isfile(filename):
                raise IOError()
            with open(filename, 'r') as f:
                self._task_variables = set(json.load(f))
        return self._task_variables

    @property
    def gene_ids(self):
        if self._gene_ids is None:
            gene_ids_file = os.path.join(self.root, 'gene_ids.json')
            if not os.path.isfile(gene_ids_file):
                if self.gene_expression_file is not None:
                    names = self.gene_expression_file['gene_names']
                else:
                    with h5py.File(self.gene_expression_path, 'r') as f:
                        names = f['gene_names']
                gene_ids = [name.decode('utf-8') for name in names]
                with open(gene_ids_file, 'w') as f:
                    json.dump(gene_ids, f)
            else:
                with open(gene_ids_file, 'r') as f:
                    gene_ids = json.load(f)
            self._gene_ids = set(gene_ids)
        return self._gene_ids

    @property
    def all_sample_ids(self):
        if self._all_sample_ids is None:
            all_sample_ids_file = os.path.join(self.root, 'all_sample_ids.json')
            if not os.path.isfile(all_sample_ids_file):
                if self.gene_expression_file is not None:
                    names = self.gene_expression_file['sample_names']
                else:
                    with h5py.File(self.gene_expression_path, 'r') as f:
                        names = f['sample_names']
                all_sample_ids = [name.decode('utf-8') for name in names]
                with open(all_sample_ids_file, 'w') as f:
                    json.dump(all_sample_ids, f)
            else:
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
        label, cancer = self.task_ids[index % self.num_classes]
        filename = self.get_processed_filename(cancer)
        dataframe = pd.read_csv(filename, sep='\t', index_col=0)
        labels = dataframe[label].dropna().astype('category')

        if self.gene_expression_file is not None:
            data = self.gene_expression_data[labels.index]
        else:
            with open(self.gene_expression_path, 'r') as f:
                data = f['expression_data'][labels.index]

        return TCGATask(data, labels.cat.codes.tolist(),
            labels.cat.categories.tolist(), transform=self.transform,
            target_transform=self.target_transform)

    def _preload_gene_expression_data(self):
        self.gene_expression_file = h5py.File(self.gene_expression_path, 'r')
        self.gene_expression_data = self.gene_expression_file['expression_data']

    def _get_task_ids(self):
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
                raw_df = pd.read_csv(filepath, sep='\t', index_col=0,
                    usecols=col_in_task_variables)#.dropna(axis=0, how='any')
                dataframe = raw_df[raw_df.index.isin(self.all_sample_ids)]
                dataframe.index = dataframe.index.map(lambda index: self.all_sample_ids[index])
                dataframe.index.names = ['index']
                dataframe = dataframe.sort_index(axis=0)
                dataframe.to_csv(processed, sep='\t')
            else:
                dataframe = pd.read_csv(processed, sep='\t', index_col=0)

            num_samples_per_label = dataframe.apply(pd.value_counts)
            min_samples_per_class = num_samples_per_label.min(axis=0)
            count_classes = num_samples_per_label.count(axis=0)
            labels = min_samples_per_class[(min_samples_per_class > self.min_samples_per_class) & (count_classes > 1)]

            task_ids.extend([(label, cancer) for label in labels.index])

        return task_ids

    def download(self):
        import gzip
        import shutil
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
            print('Downloading `{0}` using `academictorrent`...'.format(self.gene_expression_filename))
            at.get(self.gene_expression_torrent, datastore=self.root)
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
    def __init__(self, data, labels, categories, transform=None,
                 target_transform=None):
        super(TCGATask, self).__init__(transform=transform,
            target_transform=target_transform, class_transform=None)
        self.data = data
        self.labels = labels
        self.categories = categories

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = self.data[index]
        target = self.labels[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.class_transform is not None:
            sample = self.class_transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (sample, target)
