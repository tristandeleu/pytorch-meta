import os
import json
from torchvision.datasets.utils import makedir_exist_ok, check_integrity


def get_asset_path(*args):
    basedir = os.path.dirname(__file__)
    return os.path.join(basedir, 'assets', *args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not os.path.isfile(filename):
        raise IOError('{} not found'.format(filename))

    if dtype is None:
        _, dtype = os.path.splitext(filename)
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError()
    return data
