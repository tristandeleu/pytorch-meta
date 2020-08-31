import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules.module import MetaModule

class MetaEmbedding(nn.Embedding, MetaModule):
    __doc__ = nn.Embedding.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        return F.embedding(
            input, params['weight'], self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

class MetaEmbeddingBag(nn.EmbeddingBag, MetaModule):
    __doc__ = nn.EmbeddingBag.__doc__

    def forward(self, input, offsets=None, per_sample_weights=None, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        return F.embedding_bag(input, params['weight'], offsets,
                               self.max_norm, self.norm_type,
                               self.scale_grad_by_freq, self.mode, self.sparse,
                               per_sample_weights, self.include_last_offset)
