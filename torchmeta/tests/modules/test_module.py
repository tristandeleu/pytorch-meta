import pytest

import torch
import torch.nn as nn

from collections import OrderedDict
from torchmeta.modules import MetaLinear, MetaModule, MetaSequential


class MetaModel(MetaModule):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.features = MetaSequential(OrderedDict([
            ('linear1', nn.Linear(2, 3)), ('relu1', nn.ReLU()),
            ('linear2', nn.Linear(3, 5)), ('relu2', nn.ReLU())]))
        self.classifier = MetaLinear(5, 7, bias=False)

    def forward(self, inputs, params=None):
        features = self.features(inputs,
            params=self.get_subdict(params, 'features'))
        return self.classifier(features,
            params=self.get_subdict(params, 'classifier'))


def test_get_subdict_unknown_key():
    model = MetaModel()
    params = OrderedDict([('classifier.weight', torch.randn(7, 5))])

    inputs = torch.randn(11, 2)
    with pytest.warns(UserWarning):
        outputs = model(inputs, params=params)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (11, 7)


def test_get_subdict_extra_key():
    model = MetaModel()
    params = OrderedDict([
        ('features.linear1.weight', torch.randn(3, 2)),
        ('classifier.weight', torch.randn(7, 5))
    ])

    inputs = torch.randn(11, 2)
    outputs = model(inputs, params=params)

    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (11, 7)
