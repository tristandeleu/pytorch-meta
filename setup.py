from setuptools import setup, find_packages
from os import path
import sys

from io import open

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
    packages=find_packages(exclude=['data', 'contrib', 'docs', 'tests']),
    install_requires=[],
)
