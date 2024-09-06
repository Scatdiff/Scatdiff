from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np



def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        b,c,h2,w = x.shape
        # print(x.shape)
        x = rearrange(x, 'b c h w  -> b (h w) c')
        # print(x.shape)
        h = self.heads

        q = self.to_q(x)
        # print(q.shape)
        context = default(context, x)
        # print(context[0][0][0])
        # print(context.shape)
        k = self.to_k(context)
        v = self.to_v(context)
        # print(v.shape)
        

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        
        
        out = einsum('b i j, b j d -> b i d', attn, v)
        # print(out[0][0][0])
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        # print(out[0][0][0])
        out = self.to_out(out)
        # print(out[0][0][0])
        out = rearrange(out, 'b (h w) c -> b c h w', h = h2, w = w)
        # out = rearrange(x, 'b (h w) c -> b h w c', h = h2, w = w)
        # print(out.shape)
        # print("cross")
        # print(out[0][0][0][0])
        return out

