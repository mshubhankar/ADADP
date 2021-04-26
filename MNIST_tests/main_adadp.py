








'''

A code for training in a differentially private manner a fully
connected network using ADADP.
Here the method is applied to the MNIST data set.

The ADADP algorithm is described in

Koskela, A. and Honkela, A.,
Learning rate adaptation for differentially private stochastic gradient descent.
arXiv preprint arXiv:1809.03832. (2018)

This code is due to Antti Koskela (@koskeant) and is based
on a code by Mikko Heikkilä (@mixheikk).

'''










import os
import copy
import datetime
import numpy as np
import pickle
import sys
import time
import logging
from collections import OrderedDict as od
# from matplotlib import pyplot as plt
import argparse

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.autograd import Variable
import torchvision

from torchvision import datasets, transforms

import linear

import adadp
import adadp_cpu

import gaussian_moments as gm

import itertools
from types import SimpleNamespace
import px_expander
from datasets import load_datasets
from models import get_model
from sampler import get_loader

data_loc = sys.argv[1]
job_number = int(sys.argv[2])


print(torch.__version__)
print(torchvision.__version__)


randomize_data = True
batch_size = 256 # Note: overwritten by BO if used, last batch is skipped if not full size
batch_proc_size = 10 # needs to divide or => to batch size


dataset = 'Gisette'
# data_loc = 'data'
model_name = 'TLNN'

use_dp = True # dp vs non-dp model
scale_grads = True
clips = [0.001, 0.01, 0.1, 0.2, 0.5, 1, 2, 10]
clips = [clips[job_number]]
grad_norm_max = 10
noise_sigma = 4
delta = 1e-5

if model_name == 'LR':
  tol = 0.1
else:
  tol = 1.0


l_rate = 0.01

run_id = 1
iterations = 5000
stage_length = 100
repeats = 3
np.random.seed(17*run_id+3)

main_dir = dataset+'_'+model_name+'/'
if not os.path.exists(main_dir):
    os.mkdir(main_dir)

for grad_norm_max in clips:
  
  curr_dir = main_dir +'noise_'+str(noise_sigma)+'_clip_'+str(grad_norm_max)+'/'
  if not os.path.exists(curr_dir):
    os.mkdir(curr_dir)

  for r in range(repeats):  


    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      print('Using cuda')
      torch.cuda.manual_seed(11*run_id+19)
      use_cuda = True
      device = 'cuda'
    else:
      use_cuda = False
      device = 'cpu'

    data_dir = './data/'

    trainset, testset, output_dim = load_datasets(dataset, data_loc)
    input_dim = np.prod(trainset[0][0].shape)

    sampling_ratio = float(batch_size)/len(trainset)




    # moments accountant
    def update_privacy_pars(priv_pars):
      verify = False
      max_lmbd = 32
      lmbds = range(1, max_lmbd + 1)
      log_moments = []
      for lmbd in lmbds:
        log_moment = 0
        '''
        print('Here q = ' + str(priv_pars['q']))
        print('Here sigma = ' + str(priv_pars['sigma']))
        print('Here T = ' + str(priv_pars['T']))
        '''
        log_moment += gm.compute_log_moment(priv_pars['q'], priv_pars['sigma'], priv_pars['T'], lmbd, verify=verify)
        log_moments.append((lmbd, log_moment))
      priv_pars['eps'], _ = gm.get_privacy_spent(log_moments, target_delta=priv_pars['delta'])
      return priv_pars


    model, loss_function = get_model(model_name, batch_size, batch_proc_size, input_dim, output_dim, device)
    # model.train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                                     shuffle=randomize_data, num_workers=4)

    model.test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                            shuffle=randomize_data, num_workers=4)

    print('model: {}'.format(model))


    for p in model.parameters():
      if p is not None:
        if p.data.dim() == 3:
          p.data.copy_( p.data[0].clone().repeat(batch_proc_size,1,1))
        elif p.data.dim() == 2:
          p.data.copy_( p.data[0].clone().repeat(batch_proc_size,1))
    if use_cuda:
      model = model.cuda()

    #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=l_rate, momentum=0)

    if use_cuda:
      optimizer = adadp.ADADP(model.parameters())
    else:
      optimizer = adadp_cpu.ADADP(model.parameters())









    def train(model, T):

      model.train_loader = get_loader(batch_size, iterations=stage_length)
      model.train()

      for batch_idx, (data, target) in enumerate(model.train_loader(trainset)):

        optimizer.zero_grad()
        loss_tot = 0

        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
        if use_cuda:
          data, target = data.cuda(), target.cuda()


        if use_dp and scale_grads:
          cum_grads = od()
          for i,p in enumerate(model.parameters()):
            if p.requires_grad:
              if use_cuda:
                cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]),requires_grad=False).cuda()
              else:
                cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]),requires_grad=False)

        for i_batch in range(batch_size//batch_proc_size):

          data_proc = data[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size,:]
          target_proc = target[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size]

          if data_proc.shape[0] != batch_proc_size:
            print('skipped last microbatch')
            continue
          
          output = model(data_proc)
          loss = loss_function(output,target_proc)
          loss_tot += loss.data

          loss.backward()

          if use_dp and scale_grads:
            px_expander.acc_scaled_grads(model=model,C=grad_norm_max, cum_grads=cum_grads, use_cuda=use_cuda)
            optimizer.zero_grad()


        if use_dp:
          px_expander.add_noise_with_cum_grads(model=model, C=grad_norm_max, sigma=noise_sigma, cum_grads=cum_grads, use_cuda=use_cuda)

        # step1 corresponds to the first part of ADADP (i.e. only one step of size half),
        # step2 to the second part (error estimate + step size adaptation)

        if batch_idx%2 == 0:
          optimizer.step1()
        else:
          optimizer.step2(tol)

        #For SGD:
        #optimizer.step()


        T += 1

      return T, loss_tot.item()/batch_size








    def test(model):

      model.eval()

      test_loss = 0
      correct = 0

      for data, target in model.test_loader:

        if data.shape[0] != model.batch_size:
          print('skipped last batch')
          continue

        data, target = Variable(data, requires_grad=False), Variable(target, requires_grad=False)
        if use_cuda:
          data, target = data.cuda(), target.cuda()

        for i_batch in range(model.batch_size//batch_proc_size):

          data_proc = data[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size,:]
          target_proc = target[i_batch*batch_proc_size:(i_batch+1)*batch_proc_size]
          if use_cuda:
            data_proc = data_proc.cuda()
            target_proc = target_proc.cuda()

          output = model(data_proc)

          test_loss += loss_function(output, target_proc).item()

          pred = output.data.max(1, keepdim=True)[1]

          correct += pred.eq(target_proc.data.view_as(pred)).cpu().sum()

      test_loss /= len(model.test_loader.dataset)

      acc = correct.numpy() / len(model.test_loader.dataset)

      print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(model.test_loader.dataset),
        100. * acc))

      return test_loss, acc




    priv_pars = od()
    priv_pars['T'], priv_pars['eps'],priv_pars['delta'], priv_pars['sigma'], priv_pars['q'] = 0, 0, delta, noise_sigma, sampling_ratio





    # accs = []
    epsilons = []
    log_file = open(curr_dir+'log'+str(r+1)+'.txt', 'w')
    for stage in range(iterations//stage_length):


      # accs.append(acc)

      priv_pars['T'], train_loss = train(model, priv_pars['T'])



      loss, acc = test(model)
      print('[Stage %d/%d] [Training Loss: %f] [Testing Loss %f] [Testing Accuracy %f]' %
                     (stage+1, iterations/stage_length, train_loss, loss, acc))
      print('[Stage %d/%d] [Training Loss: %f] [Testing Loss %f] [Testing Accuracy %f]' %
                     (stage+1, iterations/stage_length, train_loss, loss, acc), file=log_file)
      
      print('Current privacy pars: {}'.format(priv_pars))
      if use_dp and scale_grads and noise_sigma > 0:
        update_privacy_pars(priv_pars)

      epsilons.append(priv_pars['eps'])


    # Save the test accuracies
    # np.save('accs_' +str(run_id) + '_' + str(noise_sigma) + '_' + str(batch_size),accs)
