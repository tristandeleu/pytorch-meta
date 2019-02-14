from setuptools import setup, find_packages
from os import path

from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='torchmetadatasets',
    version='1.0.0',
    description='Dataloaders for meta-learning in Pytorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/tristandeleu/pytorch-meta-dataloader',
    keywords='meta-learning pytorch',
    packages=find_packages(exclude=['data', 'contrib', 'docs', 'tests']),
    install_requires=[],
)
