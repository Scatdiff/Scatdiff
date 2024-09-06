import torch
from torch.nn import Module, Linear
import numpy as np


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        # print(ctx.device, x.device)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        # print(gate.shape)
        bias = self._hyper_bias(ctx)
        # print(bias.shape)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        # print(self._layer(x).shape)
        ret = self._layer(x) * gate + bias
        return ret
