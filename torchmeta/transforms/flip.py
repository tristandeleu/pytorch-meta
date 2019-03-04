import torchvision.transforms.functional as F

class HorizontalFlip(object):
    def __iter__(self):
        return iter([HorizontalFlip()])

    def __call__(self, image):
        return F.hflip(image)

class VerticalFlip(object):
    def __iter__(self):
        return iter([VerticalFlip()])

    def __call__(self, image):
        return F.vflip(image)
