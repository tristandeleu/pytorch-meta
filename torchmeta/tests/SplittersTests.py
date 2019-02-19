import unittest
import os
import torch
import collections

import torchmeta as tmds


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, samples_per_class):
        self.labels = []
        for label, num in enumerate(samples_per_class):
            self.labels += [label for _ in range(num)]

    def __getitem__(self, index):
        return None, self.labels[index]

    def __len__(self):
        return len(self.labels)


class TestUtil(unittest.TestCase):
    def setUp(self):
        data_dir = os.path.join('.', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        tmds.TCGA(data_dir, download=True)
        self.data_dir = data_dir

    def test_stratified_sampling(self):
        dataset = DummyDataset([1000, 2])

        lenghts = [500, 502]
        datasets = tmds.stratified_split(dataset, lenghts)

        for dataset in datasets:
            _, label = dataset.__getitem__(len(dataset) - 1)
            self.assertEquals(label, 1)

    def test_stratified_sampling_TCGA(self):
        metadataset = tmds.TCGA(self.data_dir)

        third_size = 10
        samples = 100
        loader = tmds.MetaDataLoader(metadataset, batch_size=10)
        batch = next(iter(loader))
        for dataset in batch:
            for i in range(dataset.num_classes + third_size, len(dataset) - dataset.num_classes - third_size, max(1, len(dataset) // samples)):
                length_minority_set = i

                lengths = [length_minority_set - third_size, len(dataset) - length_minority_set - third_size,2 * third_size]

                sets = tmds.stratified_split(dataset, lengths)

                all_labels = [[label for _, label in dataset] for dataset in sets]
                classes = list(set([label for labels in all_labels for label in labels]))
                contains_samples_for_all_classes = all([[class_name in label_list for label_list in all_labels] for class_name in classes])

                self.assertTrue(contains_samples_for_all_classes)

                matches_lengths = all([len(dataset) == length for dataset, length in zip(sets, lengths)])

                self.assertTrue(matches_lengths)

    def test_stratified_sampling_div_zero_edgecase(self):
        whole_dataset = DummyDataset([20, 20, 20])
        lengths = [15, 15, 15, 15]
        sets = tmds.stratified_split(whole_dataset, lengths, min_num_minority=5)

        all_labels = [[label for _, label in dataset] for dataset in sets]
        counters = [collections.Counter(labels) for labels in all_labels]

        contains_samples_for_all_classes = all([counter == {0: 5, 1: 5, 2: 5} for counter in counters])
        self.assertTrue(contains_samples_for_all_classes)

        matches_lengths = all([len(dataset) == length for dataset, length in zip(sets, lengths)])
        self.assertTrue(matches_lengths)

    def test_stratified_sampling_min_zero_edgecase(self):
        whole_dataset = DummyDataset([580, 23, 30, 19, 34, 33, 52, 19, 20])
        lengths = [574, 53, 31, 91, 27, 34]
        sets = tmds.stratified_split(whole_dataset, lengths, min_num_minority=0)

        all_labels = [[label for _, label in dataset] for dataset in sets]
        counters = [collections.Counter(labels) for labels in all_labels]

        matches_lengths = all([len(dataset) == length for dataset, length in zip(sets, lengths)])
        self.assertTrue(matches_lengths)

    def test_stratified_sampling_higher_min(self):
        whole_dataset = DummyDataset([10000, 20, 20])
        lengths = [9995, 15, 15, 15]
        sets = tmds.stratified_split(whole_dataset, lengths, min_num_minority=5)

        all_labels = [[label for _, label in dataset] for dataset in sets]
        counters = [collections.Counter(labels) for labels in all_labels]

        contains_samples_for_all_classes = all([counter == {0: 5, 1: 5, 2: 5} for counter in counters[1:]])
        self.assertTrue(contains_samples_for_all_classes)

        matches_lengths = all([len(dataset) == length for dataset, length in zip(sets, lengths)])
        self.assertTrue(matches_lengths)


if __name__ == '__main__':
    unittest.main()
