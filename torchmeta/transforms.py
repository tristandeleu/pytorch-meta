class FixedClass(object):
    def __init__(self, index):
        self.index = index

    def __call__(self, index):
        return self.index

    def __repr__(self):
        return '{0}()'.format(self.__class__.__name__)
