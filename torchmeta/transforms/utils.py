from torchvision.transforms import Compose
from torchmeta.tasks import Task

def apply_wrapper(wrapper, task_or_dataset=None, *args, **kwargs):
    if task_or_dataset is None:
        return wrapper(*args, **kwargs)

    from torchmeta.dataset import MetaDataset
    if isinstance(task_or_dataset, Task):
        wrapper_ = wrapper(*args, **kwargs)
        return wrapper_(task_or_dataset)
    elif isinstance(task_or_dataset, MetaDataset):
        if task_or_dataset.dataset_transform is None:
            dataset_transform = wrapper(*args, **kwargs)
        else:
            dataset_transform = Compose([task_or_dataset.dataset_transform,
                wrapper(*args, **kwargs)])
        task_or_dataset.dataset_transform = dataset_transform
        return task_or_dataset
    else:
        raise NotImplementedError()
