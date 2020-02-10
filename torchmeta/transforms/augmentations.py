import torchvision.transforms.functional as F

class Rotation(object):
    def __init__(self, angle, resample=False, expand=False, center=None):
        super(Rotation, self).__init__()
        if isinstance(angle, (list, tuple)):
            self._angles = angle
            self.angle = None
        else:
            self._angles = [angle]
            self.angle = angle
            if angle % 360 == 0:
                import warnings
                warnings.warn('Applying a rotation of {0} degrees (`{1}`) as a '
                    'class augmentation on a dataset is equivalent to the original '
                    'dataset.'.format(angle, self), UserWarning, stacklevel=2)

        self.resample = resample
        self.expand = expand
        self.center = center

    def __iter__(self):
        return iter(Rotation(angle, resample=self.resample, expand=self.expand,
            center=self.center) for angle in self._angles)

    def __call__(self, image):
        if self.angle is None:
            raise ValueError('The value of the angle is unspecified.')
        # QKFIX: Explicitly compute the pixel fill value due to an
        # incompatibility between Torchvision 0.5 and Pillow 7.0.0
        # https://github.com/pytorch/vision/issues/1759#issuecomment-583826810
        # Will be fixed in Torchvision 0.6
        fill = tuple([0] * len(image.getbands()))
        return F.rotate(image, self.angle % 360, self.resample,
                        self.expand, self.center, fill=fill)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        if (self.angle is None) or (other.angle is None):
            return self._angles == other._angles
        return (self.angle % 360) == (other.angle % 360)

    def __repr__(self):
        if self.angle is None:
            return 'Rotation({0})'.format(', '.join(map(str, self._angles)))
        else:
            return 'Rotation({0})'.format(self.angle % 360)

    def __str__(self):
        if self.angle is None:
            return 'Rotation({0})'.format(', '.join(map(str, self._angles)))
        else:
            return 'Rotation({0})'.format(self.angle)

class HorizontalFlip(object):
    def __iter__(self):
        return iter([HorizontalFlip()])

    def __call__(self, image):
        return F.hflip(image)

    def __repr__(self):
        return 'HorizontalFlip()'

class VerticalFlip(object):
    def __iter__(self):
        return iter([VerticalFlip()])

    def __call__(self, image):
        return F.vflip(image)

    def __repr__(self):
        return 'VerticalFlip()'
