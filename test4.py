import os
import argparse
import torch
from tqdm.auto import tqdm
from scipy.io import loadmat
from einops import rearrange, reduce

from models.autoencoder5 import *

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

parser = argparse.ArgumentParser()
parser.add_argument('--latent_dim', type=int, default=1024)  # The dimension after straightening the scattered field
parser.add_argument('--num_steps', type=int, default=200)    # Iteration time T
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
# Arguments
parser.add_argument('--ckpt', type=str, default='/DATA/zh/diffusionEMIS/logs/pre_x0_norm3_100000*loss2_batchsize_16/ckpt_1.000000_80000.pt') # Model Path
parser.add_argument('--save_dir', type=str, default='./results')
# Datasets and loaders
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

# Checkpoint

ckpt = torch.load(args.ckpt)

# Model
model = AutoEncoder(args).to(device)
model.load_state_dict(ckpt['state_dict'])

# print(model)

#--------------------------------------------------------->Mnist-Circle
bpPath = '/DATA/zh/ddpm/data/bp/BPmat_test/bp_circle0'
bppath = os.listdir(bpPath)
bppath.sort()
imgPath = '/DATA/zh/ddpm/data/bp/BPmat_test/sample_circle0'
imgpath = os.listdir(imgPath)
imgpath.sort()
escaPath = '/DATA/zh/ddpm/data/E_sca_test'
escapath = os.listdir(escaPath)
escapath.sort()
somPath = '/DATA/zh/ddpm/data/SOM_test'
sompath = os.listdir(somPath)
sompath.sort()
# --------------------------------------------------------->Mnist-Circle
#--------------------------------------------------------->EMnist-Circle
# bpPath = '/DATA/zh/ddpm/data/EMNIST-Circle/bp_letters'
# bppath = os.listdir(bpPath)
# bppath.sort()
# imgPath = '/DATA/zh/ddpm/data/EMNIST-Circle/sample_circle'
# imgpath = os.listdir(imgPath)
# imgpath.sort()
# escaPath = '/DATA/zh/ddpm/data/EMNIST-Circle/Es_letters'
# escapath = os.listdir(escaPath)
# escapath.sort()
# somPath = '/DATA/zh/ddpm/data/EMNIST-Circle/som'
# sompath = os.listdir(somPath)
# sompath.sort()
#--------------------------------------------------------->EMnist-Circle


#------------------------------------------------------------>loss
total_loss = 0

for i in tqdm(range(1000)):
    feature = loadmat(os.path.join(bpPath, bppath[i]))

    img = loadmat(os.path.join(imgPath, imgpath[i]))
    img = img['matrix']  # mnist
    # img = img['element_resized']  # emnist
    img = torch.tensor(img).to(device)

    E_sca = loadmat(os.path.join(escaPath, escapath[i]))
    E_sca = E_sca['matrix']
    real = E_sca.real
    real = (real + 2.1211)/(1.2551 + 2.1211)
    real = rearrange(real, 'b c -> (b c) 1')
    imag = E_sca.imag
    imag = (imag + 0.9784)/(1.6746 + 0.9784)
    imag = rearrange(imag, 'b c -> (b c) 1')
    E_sca =np.concatenate((real,imag), axis=-1)
    E_sca = rearrange(E_sca, 'b c ->1 b c ')
    E_sca = torch.tensor(E_sca).to(torch.float32).to(device)

    x = feature['matrix']
    x = x/2.5
    x = rearrange(x, 'b c ->1 1 b c ')  # BP diagram
    x = torch.tensor(x).to(torch.float32).to(device)

    recons = model.decode(64, E_sca, x).detach()
    # print(recons.shape)
    recons = rearrange(recons, '1 1 h w-> h w', h =64, w=64)
    generated_images = recons + 1

    #--------------------->loss1 MSEloss
    # loss = F.mse_loss(img, generated_images, reduction = 'none')
    # loss = reduce(loss, 'b ... -> b', 'mean')
    # print(loss.mean())
    # total_loss = total_loss + loss.mean()
    #--------------------->loss1 MSEloss

    #--------------------->loss2 ERRtot
    loss = torch.abs(img - generated_images)
    loss = loss/img
    loss = loss**2
    loss = reduce(loss, 'b ... -> b', 'mean')
    # print(loss.mean())
    x = torch.sqrt(loss.mean())
    print(x)
    total_loss = total_loss + x
    #--------------------->loss2 ERRtot

    #--------------------->loss3 MAPE
    # loss = torch.abs(img - generated_images)
    # loss = loss/img
    # print(loss.mean())
    # total_loss = total_loss + loss.mean()
    #--------------------->loss3 MAPE
    #--------------------->loss4 SSIM
    # img = img.to(torch.float32)
    # c1 = 0.02
    # c2 = 0.06
    # mean1 = torch.mean(img)
    # mean2 = torch.mean(generated_images)
    # std1 = torch.std(img)
    # std2 = torch.std(generated_images)
    # # cov = torch.cov(img,generated_images)
    # cov = torch.sum((img - mean1)*(generated_images - mean2))/4095
    # x = (2*mean1*mean2+c1)*(2*cov+c2)/((mean1*mean1+mean2*mean2+c1)*(std1*std1+std2*std2+c2))
    # print(x)
    # total_loss = total_loss + x
    #--------------------->loss4 SSIM


print(total_loss/1000)
#------------------------------------------->loss


