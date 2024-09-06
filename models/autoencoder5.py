import torch
from torch.nn import Module
from .diffusion5 import *     # diffusion2是原版，预测η噪声，diffusion3是直接预测x0

class AutoEncoder(Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.diffusion = DiffusionPoint(
            net = Unet(
                dim = 64,
                context_dim = 2,
                channels = 1,
                dim_mults = (1, 2, 2)
            ),
            var_sched = VarianceSchedule(
                num_steps=args.num_steps,
                beta_1=args.beta_1,
                beta_T=args.beta_T,
                mode=args.sched_mode
            )
        )



    def decode(self, num_points, Esca, bp, flexibility=0.0, ret_traj=False):       
        return self.diffusion.sample(num_points, Esca, bp, flexibility=flexibility, ret_traj=ret_traj)

    def get_loss(self, x, Esca, bp, J, J_):
        loss = self.diffusion.get_loss(x, Esca, bp, J, J_)   
        return loss
