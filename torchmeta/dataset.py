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

    def class_transform(self, class_index, image):
        transform_index = (class_index // self.num_classes) - 1
        if transform_index < 0:
            return image
        transform = self.class_transforms[transform_index]
        return transform(image)

    def get_length(self, class_index):
        raise NotImplementedError()

    @property
    def num_classes(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_classes * (len(self.class_transforms) + 1)
