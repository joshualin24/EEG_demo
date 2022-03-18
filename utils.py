"""Utility modules."""


import torch
from torch import nn
import math


class WeightNorm(nn.Module):
    """
    Weight normalization implementation insensitive to deep copying.

    Full credit to https://gist.github.com/rtqichen/b22a9c6bfc4f36e605a7b3ac1ab4122f.
    """
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            g = torch.norm(w)
            v = w/g.expand_as(w)
            g = nn.Parameter(g.data)
            v = nn.Parameter(v.data)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


class PositionalEncoding(nn.Module):
    """
    Position encoding layer for transformer-based modules.

    This implementation is adopted from the Pytorch tutorial
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos = torch.arange(max_len).unsqueeze(1)
        factor = -math.log(10000.0) / d_model
        div_even = torch.exp(torch.arange(0, d_model, 2) * factor)
        div_odd = torch.exp((torch.arange(1, d_model, 2) - 1) * factor)
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div_even)
        pe[0, :, 1::2] = torch.cos(pos * div_odd)
        self.register_buffer('pe', pe)

    def forward(self, data: torch.Tensor):
        """
        Input must have dimension `(batch, ..., sequence, embedding)`.
        """
        data = data + self.pe[:, :data.size(-2)]
        return self.dropout(data)