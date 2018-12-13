import torchvision.transforms.functional as F

class Rotation(object):
    def __init__(self, angle, resample=False, expand=False, center=None):
        super(Rotation, self).__init__()
        if isinstance(angle, (list, tuple)):
            self._angles = angle
            self.angle = None
        else:
            if angle % 360 == 0:
                # TODO: warn the transform is the identity, which already exists
                pass
            self._angles = [angle]
            self.angle = angle

        self.resample = resample
        self.expand = expand
        self.center = center

    def __iter__(self):
        return iter(Rotation(angle, resample=self.resample, expand=self.expand,
            center=self.center) for angle in self._angles)

    def __call__(self, image):
        if self.angle is None:
            raise ValueError()
        return F.rotate(image, self.angle, self.resample,
                        self.expand, self.center)
