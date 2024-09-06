import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
from torch import nn, einsum
from functools import partial
from tqdm import tqdm
import os
import math
from .common import *
from .attend import Attend
from einops.layers.torch import Rearrange
from einops import rearrange, reduce, repeat
from .attention2 import CrossAttention
# torch.set_default_tensor_type('torch.cuda.FloatTensor')

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):            
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)   
        self.norm = nn.GroupNorm(groups, dim_out)   
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock1(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, feature_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        # self.mlp2 = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(4096, dim_out * 2)
        # ) if exists(feature_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        # self.block3 = Block(2, 1, groups = 1)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        # self.block4 = Block(dim, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, feature_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            # print(time_emb.shape)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        # print(x.shape)
        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model

class VarianceSchedule(Module):

    def __init__(self, num_steps, beta_1, beta_T, mode='linear'):
        super().__init__()
        assert mode in ('linear', )
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, steps=num_steps)

        betas = torch.cat([torch.zeros([1]), betas], dim=0)     # Padding

        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i-1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas

    def last_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps+1), batch_size)
        lt = self.num_steps * (np.ones_like(ts))
        return lt.tolist()

class PointwiseNet(Module):

    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = F.leaky_relu
        self.residual = residual
        self.layers = ModuleList([
            ConcatSquashLinear(2, 128, context_dim+3),
            ConcatSquashLinear(128, 256, context_dim+3),
            ConcatSquashLinear(256, 512, context_dim+3),
            ConcatSquashLinear(512, 256, context_dim+3),
            ConcatSquashLinear(256, 128, context_dim+3),
            ConcatSquashLinear(128, 1, context_dim+3)
        ])

    def forward(self, x, beta, context, bp):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        x_0 = x
        x = torch.cat([x, bp], dim = -1)

        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)          # (B, 1, 1)
        context = context.view(batch_size, 1, -1)   # (B, 1, F)

        context = context.to(torch.float32)

        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)  # (B, 1, 3)
        ctx_emb = torch.cat([time_emb, context], dim=-1)    # (B, 1, F+3)
        # print(ctx_emb.shape)

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            # print(out.shape)
            if i < len(self.layers) - 1:
                out = self.act(out)

        # print(out.shape)
        if self.residual:
            # print((x+out).shape)
            return x_0 + out
            
        else:
            return out

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,    # defaults to full attention only for inner most layer
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # self.init_conv = nn.Conv2d(input_channels+2, init_dim, 7, padding = 3)
        # self.init_conv = nn.Conv2d(input_channels+1, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock1, groups = resnet_block_groups)

        # time embeddings          (batchsize x time_dim)

        time_dim = dim * 4
        feature_dim = 8192

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(                 
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else CrossAttention

            if(ind == 0):
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim = time_dim, feature_dim = feature_dim),
                    block_klass(dim_in, dim_in, time_emb_dim = time_dim, feature_dim = feature_dim),
                    attn_klass(dim_in, context_dim, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
                ]))
            else:
                self.downs.append(nn.ModuleList([
                    block_klass(dim_in, dim_in, time_emb_dim = time_dim, feature_dim = feature_dim),
                    block_klass(dim_in, dim_in, time_emb_dim = time_dim, feature_dim = feature_dim),
                    attn_klass(dim_in, context_dim, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, feature_dim = feature_dim)
        self.mid_attn = FullAttention(mid_dim, context_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, feature_dim = feature_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else CrossAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, feature_dim = feature_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, feature_dim = feature_dim),
                attn_klass(dim_out, context_dim, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, feature_dim = feature_dim)
        self.initial1 = block_klass(input_channels+1, init_dim, time_emb_dim = time_dim, feature_dim = feature_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, context, time, feature, x_self_cond = None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        # print(time)
        t = self.time_mlp(time)
        # print(t.shape)
        x1 = torch.cat((x,feature), axis=1)
        # print(x1.shape)
        x = self.initial1(x1, t)   # later
        # x = self.init_conv(x1)   # former
        r = x.clone()

        

        h = []


        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, feature)
            h.append(x)

            x = block2(x, t, feature)
            x = attn(x, context) + x
            h.append(x)

            x = downsample(x)
            # print("down")

        x = self.mid_block1(x, t, feature)
        x = self.mid_attn(x, context) + x
        x = self.mid_block2(x, t, feature)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, feature)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, feature)
            x = attn(x, context) + x

            x = upsample(x)
            # print("up")

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, feature)
        return self.final_conv(x)



class DiffusionPoint(Module):

    def __init__(self, net, var_sched:VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched

    def get_loss(self, x_0, context, feature, J, J_, t=None):   # predict x0
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, waves now (B, F).
        """
        batch_size, _, _, point_dim = x_0.size()
        if t == None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]
        # print(beta.shape)

        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1, 1)       # (B, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1, 1)   # (B, 1, 1)
        SRN = alpha_bar/(1-alpha_bar)
        #----------------------------------->min(SNR,5)
        # for i in range(len(SRN)):
        #     if SRN[i]>5:
        #         SRN[i]=5
        #----------------------------------->min(SNR,5)
        SRN = SRN.reshape(-1,1,1,1)
        # print(SRN.shape)

        e_rand = torch.randn_like(x_0) # (B, N, d)
        # print(e_rand.shape)

        e_theta = self.net(c0 * x_0 + c1 * e_rand, context=context, time=beta, feature=feature)

        loss = F.mse_loss((e_theta.view(-1, point_dim))*torch.sqrt(SRN), (x_0.view(-1, point_dim))*torch.sqrt(SRN), reduction='mean')

        loss2 = self.loss2(e_theta, J, J_)

        # with open("/DATA/zh/diffusionEMIS/logs/loss3.txt", 'a') as train_loss1:
        #     train_loss1.write(str(loss.item())+"\n")

        # with open("/DATA/zh/diffusionEMIS/logs/loss4.txt", 'a') as train_loss2:
        #     train_loss2.write(str(loss2.item())+"\n")

        return 100000*loss2 + loss

    def loss1(self, output, E_sca, Gs, Gd, E_inc):  # (b, 1, 32, 16)

        b, _, _, _ = output.shape
        output = rearrange(output, 'b 1 h w -> b (h w)  ')
        complex_img = np.zeros((b, 4096, 4096), dtype=complex)
        complex_img = torch.tensor(complex_img).type(torch.complex64).to(device)
        for i in range(b):
            for j in range(4096):
                complex_img[i][j][j] = output[i][j]


        I = np.identity(4096, dtype=complex)
        I = torch.tensor(I).type(torch.complex64).to(device)
        I = I.repeat(b,1,1)

        result = I - complex_img@Gd
        x_verse = torch.linalg.inv(result.cpu()).cuda()
        # x_verse = result

        result2 = complex_img@E_inc
        result = Gs@x_verse@result2

        result = abs(result)
        loss = F.mse_loss(E_sca, result, reduction='mean')
        # loss = reduce(loss, 'b ... -> b', 'mean')

        return loss

    def loss2(self, output, J, J_):
        b, _, _, _ = output.shape
        output = rearrange(output, 'b 1 h w -> b (h w) 1 ')

        J_ = output * J_

        loss = F.mse_loss(J_, J, reduction = 'mean')

        return loss


    def sample(self, num_points, context, feature, point_dim=64, flexibility=0.0, ret_traj=False):
        batch_size = context.size(0)
        t=1
        x_T = torch.randn([batch_size, 1, num_points, point_dim]).to(context.device)
        beta = self.var_sched.betas[[t]*batch_size]
        e_theta = self.net(x_T, context=context, time=beta, feature=feature)
        return e_theta

