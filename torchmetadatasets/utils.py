import torch.utils.data as data

class Transform_setter:
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, dataset):
        dataset.transform = self.transform

def recursive_apply(datasets, function):
    for dataset in datasets:
        _recursive_apply(dataset, function)

def _recursive_apply(dataset, function):
    if isinstance(dataset, data.ConcatDataset):
        for element in dataset.datasets:
            _recursive_apply(element, function)
    elif isinstance(dataset, data.Subset):
        _recursive_apply(dataset.dataset, function)
    else:
        function(dataset)