import argparse
from ast import arg
from curses import init_pair
import os
import numpy as np
import math
import sys
import utils

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
# from models import WGAN_EA
from models import WGAN_GFV
from models import Completion_EA as premodel
# from dataset import ShapeNetGFV
from dataset import ShapeNet
# import LoadNpy
import logging
from tqdm import tqdm
import time
from visualization import plot_pcd_one_view
from metrics.metric import l1_cd
from metrics.loss import cd_loss_L1, emd_loss
import copy
from torch.autograd import Variable
os.makedirs("images", exist_ok=True)


def getLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关
    formatter = logging.Formatter(fmt="[%(asctime)s|%(filename)s|%(levelname)s] %(message)s",
                                  datefmt="%a %b %d %H:%M:%S %Y")
    # StreamHandler
    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    # FileHandler
    work_dir = os.path.join("GanModel/logs",
                            time.strftime("%Y-%m-%d-%H.%M", time.localtime()))  # 日志文件写入目录
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    fHandler = logging.FileHandler(work_dir + '/log.txt', mode='w')
    fHandler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关
    fHandler.setFormatter(formatter)  # 定义handler的输出格式
    logger.addHandler(fHandler)  # 将logger添加到handler里面

    return logger


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001,
                    help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=2560,
                    help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int,
                    default=10, help="interval betwen image samples")
# parser.add_argument('--data', metavar='DIR', default='/home/featurize/Stability-point-recovery-master/gfvdata_test/',
#                     help='Path to Complete Point Cloud Data Set')
parser.add_argument('--data', metavar='DIR', default='/home/featurize/Stability-point-recovery-master/gfvdata_test/',
                    help='Path to Complete Point Cloud Data Set')
parser.add_argument('--category', type=str, default='all', help='Category of global feature')
parser.add_argument('--split_value', default=0.9, help='Ratio of train and test data split')
parser.add_argument('--ckpt_path', type=str, default='flatten/ckpt', help='The path of pretrained model')
parser.add_argument('--save_img', type=str, default='flatten/ckpt/imgs', help='The path of pretrained model')
parser.add_argument('--num_workers', type=int, default=16)
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
parser.add_argument('--pretrained', default='/home/featurize/Stability-point-recovery-master/log/Transformer_point/all/checkpoints/model_best.pth.tar',
                    help='Use Pretrained Model for testing or resuming training')  # TODO

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Loss weight for gradient penalty
lambda_gp = 10

# load decoder

network_data = torch.load(opt.pretrained)

model_decoder = premodel.PreModel()
model_decoder.Encoder.load_state_dict(network_data['state_dict_encoder'])
model_decoder.Decoder.load_state_dict(network_data['state_dict_decoder'])
model_encoder = copy.deepcopy(model_decoder.Encoder)
model_encoder.cuda()
model_decoder.cuda()

# Initialize generator and discriminator
# generator = WGAN_GFV.Generator()
generator = copy.deepcopy(model_decoder.Decoder.mlp)
discriminator = WGAN_GFV.Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

for param in generator.parameters():
    param.requires_grad = True

for param in model_encoder.parameters():
    param.requires_grad = False

# train_dataset = ShapeNetGFV(opt.data, 'train', opt.category)
train_dataset = ShapeNet('/home/featurize/data/PCN', 'train', opt.category)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size,
                                           num_workers=opt.num_workers,
                                           shuffle=True,
                                           pin_memory=True)


# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
schedulerD = torch.optim.lr_scheduler.StepLR(
    optimizer_D, step_size=40, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(
    optimizer_G, step_size=40, gamma=0.2)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
logger = getLogger()


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.contiguous().view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ----------
#  Training
# ----------
def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
    return pc[idx[:n]]


# training save best
best_generator = 1e8
real_label = 1
fake_label = 0
batch_size = opt.batch_size
label = torch.FloatTensor(batch_size)
criterion = torch.nn.BCEWithLogitsLoss().cpu()
for epoch in range(opt.n_epochs):

    for step, (gfv, complete) in tqdm(enumerate(train_loader, 0),
                                   total=len(train_loader), smoothing=0.9):

        b, _, _ = gfv.shape

        label.resize_([b, 1]).fill_(real_label)

        # partial = gfv.cuda()
        real = complete.cuda()

        # partial = model_decoder.Decoder.mlp(partial).reshape(-1, 1024, 3)
        real = model_decoder.Decoder.mlp(real).reshape(-1, 1024, 3)
        input = gfv.cuda()
        input = Variable(input, requires_grad=True)
        real = real.data.cuda()
        label = label.cuda()

        generator = generator.train()

        discriminator = discriminator.train()
        ############################
        # (2) Update D network
        ###########################
        # if i % opt.n_critic == 0:
        discriminator.zero_grad()
        # output = discriminator(real)
        real_validity = discriminator(real)

        # errD_real = criterion(output, label)
        # errD_real.backward()

        fake_gfv = generator(input).reshape(-1, 1024, 3)

        label.data.fill_(fake_label)

        # output = discriminator(fake_gfv.detach())
        fake_validity = discriminator(fake_gfv.detach())

        gradient_penalty = compute_gradient_penalty(
                        discriminator, real.data, fake_gfv.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

        # errD_fake = criterion(output, label)

        # errD_fake.backward()
        d_loss.backward()
        # errD = errD_real + errD_fake
        optimizer_D.step()
        ############################
        # (3) Update G network
        ###########################
        if step % opt.n_critic == 0:
            alpha = 0.01
            beta = 20
            generator.zero_grad()
    #         label.data.fill_(real_label)

    #         output = discriminator(input)
    #         errG_D_x_real = criterion(output, label)
    #         errG_D_x_real.backward()
    #         output = discriminator(fake_gfv.detach())
    #         errG_D_Gx_real = criterion(output, label)
    #         errG_D_Gx_real.backward()

            # l1_loss = torch.sum(torch.abs(real - fake_gfv))
            # l1_loss.backward()

            # errG = (errG_D_x_real + errG_D_Gx_real) + alpha * l1_loss
            # errG.backward()
            # fake_gfv = generator(input)
            fake_gfv = generator(input).reshape(-1, 1024, 3)
            loss1 = emd_loss(fake_gfv, real)
            loss2 = cd_loss_L1(fake_gfv, real)
            loss = alpha * loss1 + loss2
            loss.backward()

            optimizer_G.step()
        if step % 20 == 0:
            if not os.path.exists(opt.ckpt_path):
                os.makedirs(opt.ckpt_path)
            if not os.path.exists(opt.save_img):
                os.makedirs(opt.save_img)
            plot_pcd_one_view(os.path.join(opt.save_img, 'epoch_{:03d}_{:03d}.png'.format(epoch, step)),
                              [fake_gfv[0].detach().cpu().numpy(), real[0].detach().cpu().numpy()],
                              ['coarse_fake', 'coarse_real'],
                              xlim=(-0.35, 0.35), ylim=(-0.35, 0.35),
                              zlim=(-0.35, 0.35))

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
              % (epoch, opt.n_epochs, step, len(train_loader),
                 d_loss.data, loss.data))
        # print('[%d/%d][%d/%d] Loss_G: %.4f'
        #       % (epoch, opt.n_epochs, i, len(train_loader), loss.data))

        f = open('loss_GanTrainer.txt', 'a')
        f.write('\n'+'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                % (epoch, opt.n_epochs, step, len(train_loader),
                    d_loss.data, loss.data))
        # f.write('\n'+'[%d/%d][%d/%d] Loss_G: %.4f'
        #         % (epoch, opt.n_epochs, i, len(train_loader), loss.data))

        f.close()
        # schedulerD.step()
        schedulerG.step()

    if not os.path.exists(opt.ckpt_path):
        os.makedirs(opt.ckpt_path)
    if not os.path.exists(opt.save_img):
        os.makedirs(opt.save_img)
    if epoch % 1 == 0:
        torch.save({'epoch': epoch+1,
                    'state_dict': generator.state_dict()},
                   os.path.join(opt.ckpt_path, 'generator_model_best.pth.tar'))
        torch.save({'epoch': epoch+1,
                    'state_dict': discriminator.state_dict()},
                   os.path.join(opt.ckpt_path,
                                'discriminator_model_best.pth.tar'))
        plot_pcd_one_view(os.path.join(opt.save_img, 'epoch_{:03d}.png'.format(epoch)),
                              [fake_gfv[0].detach().cpu().numpy(), complete[0].detach().cpu().numpy()],
                              ['coarse_fake', 'coarse_real'],
                              xlim=(-0.35, 0.35), ylim=(-0.35, 0.35),
                              zlim=(-0.35, 0.35))

