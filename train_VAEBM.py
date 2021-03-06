# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for VAEBM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
'''Code for training VAEBM'''

from cmath import exp
import random
import argparse
import torch
import numpy as np
import os
from torch import autograd
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.autograd import Variable
from torch import optim
# from apex import amp

from nvae_model import AutoEncoder
from train_VAEBM_distributed import init_processes
import utils
import datasets
import torchvision
from tqdm import tqdm
from ebm_models import EBM_CelebA64, EBM_LSUN64, EBM_CIFAR32, EBM_CelebA256
from thirdparty.igebm_utils import sample_data, clip_grad

from torch.utils.tensorboard import SummaryWriter

import math

def set_bn(model, bn_eval_mode, num_samples=1, t=1.0, iter=100):
    if bn_eval_mode:
        model.eval()
    else:
        model.train()

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

class SampleBuffer:
    def __init__(self, num_block, max_samples, device = torch.device('cuda:0')):
        self.max_samples = max_samples
        self.num_block = num_block
        self.buffer = [[] for _ in range(num_block)]  #each group of latent variable is a list
        self.device = device

    def __len__(self):
        return len(self.buffer[0]) #len of the buffer should be the length of list for each group of latent

    def push(self, z_list): #samples is a list of torch tensor
        for i in range(self.num_block):
            zi = z_list[i]
            zi = zi.detach().to('cpu')
            for sample in zip(zi):
                self.buffer[i].append(sample[0])
                if len(self.buffer[i]) > self.max_samples:
                    self.buffer[i].pop(0)

    def get(self, n_samples):
        sample_idx = random.sample(range(len(self.buffer[0])), n_samples)
        z_list = []
        for i in range(self.num_block):
            samples = [self.buffer[i][j] for j in sample_idx]
            samples = torch.stack(samples, 0)
            samples = samples.to(self.device)
            z_list.append(samples)

        return z_list
    def save(self,fname):
        torch.save(self.buffer,fname)

def sample_buffer(buffer, z_list_exampler, batch_size=64, t = 1, p=0.95, device=torch.device('cuda:0')):
    if len(buffer) < 1:       
        
        eps_z = [torch.Tensor(batch_size, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).cuda() \
                 for zi in z_list_exampler]

        return eps_z
    

    n_replay = (np.random.rand(batch_size) < p).sum()
    
    if n_replay > 0:
    
        eps_z_replay = buffer.get(n_replay)
        eps_z_prior = [torch.Tensor(batch_size - n_replay, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).cuda()\
                for zi in z_list_exampler]

        eps_z_combine = [torch.cat([z1,z2], dim = 0) for z1,z2 in zip(eps_z_replay, eps_z_prior)]
        
        return eps_z_combine
    else:
        eps_z = [torch.Tensor(batch_size - n_replay, zi.size(1), zi.size(2), zi.size(3)).normal_(0, 1.).cuda() \
                for zi in z_list_exampler]

    
        return eps_z

def train(model, VAE, t, train_loader, valid_loader, opt, model_path, num_classes):
    step_size = opt.step_size
    sample_step = opt.num_steps
    
    requires_grad(VAE.parameters(), False)
    train_loader = tqdm(enumerate(sample_data(train_loader)))
    

    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=opt.lr, betas = (0.99,0.999), weight_decay = opt.wd)
    
    if opt.use_amp:
        [model, VAE], optimizer = amp.initialize([model,VAE], optimizer, opt_level='O1')


    d_s_t = []
    valid_accuracies = []
    
    with torch.no_grad(): #get a bunch of samples to know how many groups of latent variables are there
        _, z_list, _ = VAE.sample(opt.batch_size, t) 
    
    num_block = len(z_list)
    
    
    if opt.use_buffer:
        buffer = SampleBuffer(num_block = num_block, max_samples = opt.buffer_size)

    noise_list = [torch.randn(zi.size()).cuda() for zi in z_list]

    # BCEWithLogitsLoss for classification
    criterion = torch.nn.BCEWithLogitsLoss()

    writer = SummaryWriter()

    
    for idx, (image, label) in train_loader:
        image = image.cuda()
        label = label.cuda()
        label = torch.nn.functional.one_hot(label, num_classes).float()
        
        noise_x = torch.randn(image.size()).cuda()
        
        if opt.use_buffer:
            #annealing the probability of sampling from buffer
            buffer_prob = min(opt.max_p, opt.max_p*idx/opt.anneal_step)
            eps_z_nograd = sample_buffer(buffer, z_list, batch_size = image.size(0), p=buffer_prob)
            eps_z = [Variable(eps_zi, requires_grad=True) for eps_zi in eps_z_nograd]
        else:
            eps_z = [Variable(torch.Tensor(zi.size()).normal_(0, 1.).cuda() , requires_grad=True) for zi in z_list]

        eps_x = torch.Tensor(image.size()).normal_(0, 1.).cuda()   

        eps_x = Variable(eps_x, requires_grad = True)


        requires_grad(parameters, False)
        
        model.eval()
        VAE.eval()

        for k in range(sample_step):
            
            logits, _, log_p_total = VAE.sample(opt.batch_size, t, eps_z)
            output = VAE.decoder_output(logits)
            neg_x = output.sample(eps_x) 
            
            log_pxgz = output.log_prob(neg_x).sum(dim = [1,2,3])
        
            #compute energy
            model_output_energy = -torch.logsumexp(model(neg_x), dim=1)
            dvalue = model_output_energy - log_p_total - log_pxgz 
            # dvalue = model(neg_x) - log_p_total - log_pxgz 
            dvalue = dvalue.mean()
            dvalue.backward()

            for i in range(len(eps_z)):    
                #update z group by group
                noise_list[i].normal_(0, 1)

                eps_z[i].data.add_(-0.5*step_size, eps_z[i].grad.data * opt.batch_size )

                eps_z[i].data.add_(np.sqrt(step_size), noise_list[i].data)    
                eps_z[i].grad.detach_()
                eps_z[i].grad.zero_()
            
            #update x
            noise_x.normal_(0, 1)
            eps_x.data.add_(-0.5*step_size, eps_x.grad.data * opt.batch_size)
            eps_x.data.add_(np.sqrt(step_size), noise_x.data)
            eps_x.grad.detach_()
            eps_x.grad.zero_()

        
        eps_z = [eps_zi.detach() for eps_zi in eps_z]

        eps_x = eps_x.detach()
        
        requires_grad(parameters, True)
        model.train()

        model.zero_grad()
        logits, _, _ = VAE.sample(opt.batch_size, t, eps_z)
        output = VAE.decoder_output(logits)
        
        if opt.use_mu_cd:
            neg_x = 0.5*output.dist.mu + 0.5
        else: 
            neg_x = output.sample(eps_x)

        # VAEBM assumes that the distribution is of the form of 
        # \propto exp(-f(x)): gradient descent update rule must
        # be f_pos - f_neg. The JEM assumes the form to be the
        # opposite: \propto exp(f(x)). We use the former's notation
        
        # EBM Loss
        pos_out = -torch.logsumexp(model(image), dim=1)
        neg_out = -torch.logsumexp(model(neg_x), dim=1)
        # pos_out = model(image)
        # neg_out = model(neg_x)
        
        # Norm loss
        norm_loss = model.spectral_norm_parallel()
        loss_reg_s = opt.alpha_s * norm_loss

        # Classification Loss
        # Flip the sign and use BCEWithLogitsLoss 
        logits = model(image)
        # logits = model.classify(image)
        classification_loss = criterion(logits, label)

        loss = pos_out.mean() - neg_out.mean()
        # print("Losses:", loss.item(), classification_loss.item(), loss_reg_s.item())
        factor = 1e-3 * math.exp(-5 * pow(1 - min(idx / 4000, 1), 2))
        loss_total = factor * loss + classification_loss + loss_reg_s
        
        if opt.use_amp:
            with amp.scale_loss(loss_total, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss_total.backward()
        
        if opt.grad_clip:
            clip_grad(model.parameters(), optimizer)

        optimizer.step()

        
        if opt.use_buffer:
            buffer.push(eps_z)

        train_loader.set_description(f'Total Loss: {loss_total.item():.5f}')
        d_s_t.append(loss_total.item())

        writer.add_scalar('Train/LossTotal', loss_total.item(), idx)
        writer.add_scalar('Train/LossSampling', loss.item(), idx)
        writer.add_scalar('Train/LossClassification', classification_loss.item(), idx)
        writer.add_scalar('Train/LossReg', loss_reg_s.item(), idx)

        if idx % 100 == 0:
            neg_img = 0.5*output.dist.mu + 0.5
            torchvision.utils.save_image(
                neg_img,
                model_path + '/images/sample_iter_{}.png'.format(idx),
                nrow=16,
                normalize=True
            )

            torch.save(d_s_t, model_path + 'd_s_t')

            valid_acc = run_validation(valid_loader, model)
            writer.add_scalar('Validation/Accuracy', valid_acc, idx)
            valid_accuracies.append(valid_acc)

            torch.save(valid_accuracies, model_path + 'valid_accuracies')

        
        if idx % 500 == 0:
            state_dict = {}
            state_dict['model'] = model.state_dict()
            state_dict['optimizer'] = optimizer.state_dict()
            torch.save(state_dict, model_path + 'EBM_{}.pth'.format(idx))

        if idx == opt.total_iter:
            break

def run_validation(valid_loader, model):
    correct_count = 0
    total_count = 0
    model.eval()
    for _, (image, label) in enumerate(valid_loader):
        image = image.cuda()
        label = label.cuda()
        prediction = model(image)
        prediction = torch.argmax(prediction, dim=1)
        correct_count += torch.sum(prediction == label)
        total_count += len(label)
    return correct_count / total_count

def main(eval_args):
    # ensures that weight initializations are all the same
    logging = utils.Logger(eval_args.local_rank, eval_args.save)

    # load a checkpoint
    logging.info('loading the model at:')
    logging.info(eval_args.checkpoint)
    checkpoint = torch.load(eval_args.checkpoint, map_location='cpu')
    args = checkpoint['args']

    logging.info('loaded model at epoch %d', checkpoint['epoch'])
    if not hasattr(args, 'ada_groups'):
        logging.info('old model, no ada groups was found.')
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        logging.info('old model, no min_groups_per_scale was found.')
        args.min_groups_per_scale = 1

    arch_instance = utils.get_arch_cells(args.arch_instance)
    
    #define and load pre-trained VAE
    model = AutoEncoder(args, None, arch_instance)
    model = model.cuda()
    print('num conv layers:', len(model.all_conv_layers))
    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    
    t = 1 #temperature of VAE samples
    train_loader, valid_loader, num_classes = datasets.get_loaders(eval_args)

    if eval_args.dataset == 'cifar10':
        EBM_model = EBM_CIFAR32(num_classes, 3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    elif eval_args.dataset == 'celeba_64':
        EBM_model = EBM_CelebA64(3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    elif eval_args.dataset == 'lsun_church':
        EBM_model = EBM_LSUN64(3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    elif eval_args.dataset == 'celeba_256':
        EBM_model = EBM_CelebA256(3,eval_args.n_channel, data_init = eval_args.data_init).cuda()
    else:
        raise Exception("choose dataset in ['cifar10', 'celeba_64', 'lsun_church', 'celeba_256']")
    
    
    model_path = './saved_models/{}/{}/'.format(eval_args.dataset, eval_args.experiment)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        os.makedirs(model_path + '/images/')
    
    #use 5 batch of training images to initialize the data dependent init for weight norm
    init_image = []
    for idx, (image) in enumerate(train_loader):
        img = image[0]
        init_image.append(img)
        if idx == 4:
            break
    init_image = torch.stack(init_image).cuda()
    init_image = init_image.view(-1,3,eval_args.im_size,eval_args.im_size)

    EBM_model(init_image) #for initialization
    
    
    #call the training function
    train(EBM_model, model, t, train_loader, valid_loader, eval_args, model_path, num_classes)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('training of VAEBM')
    # experimental results
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/cifar10/checkpoint.pt',
                        help='location of the NVAE checkpoint')
    parser.add_argument('--experiment', default='EBM_1', help='experiment name, model chekcpoint and samples will be saved here')

    parser.add_argument('--save', type=str, default='./log',
                        help='location of the NVAE logging')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='which dataset to use')
    parser.add_argument('--im_size', type=int, default=32, help='size of image')

    parser.add_argument('--data', type=str, default='../data/',
                        help='location of the data file')

    parser.add_argument('--lr', type=float, default=4e-5,
                        help='learning rate for EBM')

    # DDP.
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process')
    parser.add_argument('--world_size', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    
    parser.add_argument('--batch_size', type=int, default = 32, help='batch size for training EBM')
    parser.add_argument('--n_channel', type=int, default = 128, help='initial number of channels of EBM')

    # traning parameters
    parser.add_argument('--alpha_s', type=float, default=0.2, help='spectral reg coef')

    parser.add_argument('--step_size', type=float, default=8e-5, help='step size for LD')
    parser.add_argument('--num_steps', type=int, default=10, help='number of LD steps')
    parser.add_argument('--total_iter', type=int, default=30000, help='number of training iteration')


    parser.add_argument('--wd', type=float, default=3e-5, help='weight decay for adam')
    parser.add_argument('--data_init', dest='data_init', action='store_false', help='data depedent init for weight norm')
    parser.add_argument('--use_mu_cd', dest='use_mu_cd', action='store_true', help='use mean or sample from the decoder to compute CD loss')
    parser.add_argument('--grad_clip', dest='grad_clip', action='store_false',help='clip the gradient as in Du et al.')    
    parser.add_argument('--use_amp', dest='use_amp', action='store_true', help='use mix precision')
    
    #buffer
    parser.add_argument('--use_buffer', dest='use_buffer', action='store_true', help='use persistent training, default is false')
    parser.add_argument('--buffer_size', type=int, default = 10000, help='size of buffer')
    parser.add_argument('--max_p', type=float, default=0.6, help='maximum p of sampling from buffer')
    parser.add_argument('--anneal_step', type=float, default=5000., help='p annealing step')
    
    parser.add_argument('--comment', default='', type=str, help='some comments')
    
    args = parser.parse_args()
    
    args.distributed = False
    init_processes(0, 1, main, args)