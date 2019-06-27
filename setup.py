from setuptools import setup, find_packages
from os import path
import sys

from io import open

extras = {
    'tcga': ['pandas~=0.24.0', 'academictorrents~=2.1.0', 'six~=1.11.0'],
}

here = path.abspath(path.dirname(__file__))

sys.path.insert(0, path.join(here, 'torchmeta'))
from version import VERSION

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='torchmeta',
    version=VERSION,
    description='Dataloaders for meta-learning in Pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tristandeleu/pytorch-meta-dataloader',
    keywords='meta-learning pytorch',
    packages=find_packages(exclude=['data', 'contrib', 'docs', 'tests', 'examples']),
    install_requires=[
        'torch>=1.1.0',
        'torchvision>=0.3.0',
        'numpy>=1.14.0',
        'Pillow>=5.0.0',
        'h5py~=2.9.0',
        'tqdm>=4.0.0',
    ],
    extras_requires=extras,
)
