import os
import json
import h5py
import pandas as pd

from torchmeta.dataset import MetaDataset, Task

class classproperty(property):
    """Subclass property to make classmethod properties possible"""
    def __get__(self, cls, owner):
        return self.fget.__get__(None, owner)()

class TCGA(MetaDataset):
    folder = 'tcga'
    clinical_matrix_url = 'https://tcga.xenahubs.net/download/TCGA.{0}.sampleMap/{0}_clinicalMatrix.gz'
    clinical_matrix_filename, _ = os.path.splitext(os.path.basename(clinical_matrix_url))
    gene_expression_filename = 'TCGA_HiSeqV2.hdf5'
    gene_expression_torrent = 'e4081b995625f9fc599ad860138acf7b6eb1cf6f'
    
    _task_variables = None
    _cancers = None

    def __init__(self, root, meta_train=True, min_samples_per_class=3, transform=None,
                 target_transform=None, dataset_transform=None, download=False, preload=True):
        super(TCGA, self).__init__(dataset_transform=dataset_transform)
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.meta_train = meta_train
        self.min_samples_per_class = min_samples_per_class

        self.transform = transform
        self.target_transform = target_transform
        
        self._all_sample_ids = None
        self._gene_ids = None

        if download:
            self.download()

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
    
    @classproperty
    @classmethod
    def assets_path(cls):
        cls._assets_path = os.path.join(os.path.dirname(__file__), 'assets', cls.folder)
        return cls._assets_path 

    @classproperty
    @classmethod
    def cancers(cls):
        if cls._cancers is None:
            filename = os.path.join(cls.assets_path, 'cancers.json')
            if not os.path.isfile(filename):
                raise IOError()
            with open(filename, 'r') as f:
                cls._cancers = json.load(f)
        return tuple(cls._cancers)

    @classproperty
    @classmethod
    def task_variables(cls):
        if cls._task_variables is None:
            filename = os.path.join(cls.assets_path, 'task_variables.json')
            if not os.path.isfile(filename):
                raise IOError()
            with open(filename, 'r') as f:
                cls._task_variables = set(json.load(f))
        return tuple(cls._task_variables)

    @property
    def gene_ids(self):
        if self._gene_ids is None:
            gene_ids_file = os.path.join(self.root, 'gene_ids.json')
            if not os.path.isfile(gene_ids_file):
                if self.gene_expression_file is not None:
                    names = self.gene_expression_file['gene_ids']
                else:
                    with h5py.File(self.gene_expression_path, 'r') as f:
                        names = f['gene_ids']
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
        label, cancer = self.task_ids[index]
        filename = self.get_processed_filename(cancer)
        dataframe = pd.read_csv(filename, sep='\t', index_col=0)
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

        #gene_expression_file = os.path.join(self.root, self.gene_expression_filename)
        #if not os.path.isfile(gene_expression_file):
        #    import academictorrents as at
        #    print('Downloading `{0}` using `academictorrent`...'.format(self.gene_expression_filename))
        #    at.get(self.gene_expression_torrent, datastore=self.root)
        #    print('Done')
        import academictorrents as at
        gene_expression_file = os.path.join(self.root, self.gene_expression_filename)
        print('Downloading `{0}` using `academictorrent`...'.format(self.gene_expression_filename))
        csv_file = at.get(self.gene_expression_torrent, datastore=self.root)
        if not os.path.isfile(gene_expression_file) and os.path.isfile(csv_file):
            print("Downloaded to: " + csv_file)
            print("Converting TCGA CSV dataset to HDF5. This only happens on first run.")
            df = pd.read_csv(csv_file, compression="gzip", sep="\t")
            df = df.transpose()
            df.columns = df.iloc[0]
            df = df.drop(df.index[0])
            df = df.astype(float)
            gene_ids = df.columns.values
            all_sample_ids = df.index.values
            # with open(gene_ids_file, "w") as text_file:
            #     for gene_id in gene_ids:
            #         text_file.write('{}\n'.format(gene_id))
            # with open(all_sample_ids_file, "w") as text_file:
            #     for sample_id in all_sample_ids:
            #         text_file.write('{}\n'.format(sample_id))

            f = h5py.File(gene_expression_file)
            f.create_dataset("expression_data", data=df.values)
            f.create_dataset("gene_ids", data=gene_ids.astype('S20'))
            f.create_dataset("sample_names", data=all_sample_ids.astype('S20'))
            f.close()
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
        root = os.path.join(os.path.expanduser(root), 'tcga')

        clinical_matrix_url = 'https://tcga.xenahubs.net/download/TCGA.{0}.sampleMap/{0}_clinicalMatrix.gz'
        clinical_matrix_filename, _ = os.path.splitext(os.path.basename(clinical_matrix_url))
        gene_expression_filename = 'TCGA_HiSeqV2.hdf5'
        gene_filepath = os.path.join(root, gene_expression_filename)
        if not os.path.isfile(gene_filepath):
            raise IOError()
        
        label, cancer = task_id

        processed_folder = os.path.join(root, 'clinicalMatrices', 'processed')
        filename = '{0}.tsv'.format(clinical_matrix_filename.format(cancer))
        filepath = os.path.join(processed_folder, filename)
        if not os.path.isfile(filepath):
            raise IOError()

        dataframe = pd.read_csv(filepath, sep='\t', index_col=0)
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
