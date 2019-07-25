# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import time
import math
import random
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as dist
from dataPrepare import *

torch.manual_seed(0)

MAX_LENGTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')
Training_generator, Test, Valid, WholeSet= get_dataloader()


class NNPred(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,batch_size, dropout=0.05):
        super(NNPred, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.in2lstm = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False,batch_first=True,dropout =0.1)
        self.in2bilstm = nn.Linear(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size//2,num_layers=self.num_layers,bidirectional=True,batch_first=True,dropout =0.1)
    
        self.fc0 = nn.Linear(256,128)
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64 ,output_size)
        self.in2out = nn.Linear(input_size, 64)
        self.tanh = nn.Tanh()

        
    def forward(self, input):
        #input = tensor shape[batchsize, len, num_features]
 
        bilstm_out,_= self.lstm(self.in2bilstm(input))
        
        lstm_out,_= self.lstm(self.in2lstm(input))
        out = self.tanh(self.fc0(lstm_out+bilstm_out))
        out = self.tanh(self.fc1(out))
        out =  out + self.in2out(input)
        output = self.fc2(out)# range [0 -> 1 ]
        return output

def trainIters(encoder, n_iters, print_every=1000, plot_every=1, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    #criterion = nn.SmoothL1Loss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    pltcount = 0
    prtcount = 0
    cp = 0
    for iter in range(1, n_iters + 1):
        if iter%50==1:
            cp = cp+1
            torch.save(encoder.state_dict(), str(cp)+'checkpoint.pth.tar')
        for local_batch, local_labels in Training_generator:
            if local_batch.shape[0]!=32:
                continue
            pltcount = pltcount+1
            prtcount = prtcount+1
            encoder.zero_grad()
            
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)
            
            predY = encoder(local_batch)
            loss = criterion(predY[:,-30:,:],local_labels[:,-30:,:]).to(device)
            loss.backward()
            encoder_optimizer.step()
            
            ls =  loss.detach().item()
            print_loss_total += ls
            plot_loss_total += ls
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / prtcount
            print_loss_total = 0
            prtcount = 0
            print('%s (%d %d%%) %f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / pltcount
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            pltcount = 0
    return plot_losses

def Eval_net(encoder):
    count = 0
    for local_batch, local_labels in Training_generator:
        if local_batch.shape[0]!=32:
            continue
        count = count + 1
        local_batch = local_batch.to(device)
        local_labels = local_labels.to(device)
        predY = encoder(local_batch)
        print(WholeSet.std.repeat(32,100,1).shape)
        std = WholeSet.std.repeat(32,100,1)
        std = std[:,:,:4].to(device)
        mn = WholeSet.mn.repeat(32,100,1)
        mn = mn[:,:,:4].to(device)
        rg = WholeSet.range.repeat(32,100,1)
        rg = rg[:,:,:4].to(device)
        predY = (predY*(rg*std)+mn).detach().cpu()
        pY = np.array(predY )
        local_labels = (local_labels*(rg*std)+mn).detach().cpu()
        Y = np.array(local_labels)
        pY[:,:-30,:] = Y[:,:-30,:]
        for i in range(32):
            plt.figure(i)
            plt.xlim(0,80)
            plt.ylim(0,2000)
            plt.plot(pY[i,:,2],pY[i,:,3],'r')
            plt.plot(Y[i,:,2],Y[i,:,3],'g')
        plt.show()

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()

train_iter = iter(Training_generator)
x, y = train_iter.next()
print(x.shape)
hidden_size = 256
Prednet = NNPred(x.shape[2], y.shape[2],hidden_size, 32)

print(device)

TRAN_TAG = True
if TRAN_TAG:
    if path.exists("checkpoint.pth.tar"):
        Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    plot_losses = trainIters(Prednet, 200, print_every=2)
    torch.save(Prednet.state_dict(), 'checkpoint.pth.tar')
    showPlot(plot_losses)
else:
    Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    Prednet.eval()
    Eval_net(Prednet)
