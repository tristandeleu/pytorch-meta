class Dataset(object):
    def __init__(self, class_transforms=None):
        if class_transforms is not None:
            if not isinstance(class_transforms, list):
                raise ValueError()
            class_transforms = [transform for class_transform
                in class_transforms for transform in class_transform]
        else:
            class_transforms = []
        self.class_transforms = class_transforms

    def class_transform(self, index):
        transform_index = (index // self.num_classes) - 1
        if transform_index < 0:
            return None
        return self.class_transforms[transform_index]

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_classes * (len(self.class_transforms) + 1)

class Task(object):
    def __init__(self, transform=None, target_transform=None,
                 class_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.class_transform = class_transform

    def __getitem__(self, index):
        raise NotImplementedError()
