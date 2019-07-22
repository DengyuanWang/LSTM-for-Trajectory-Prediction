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

from dataPrepare import *

torch.manual_seed(0)

MAX_LENGTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')
Training_generator, Test, Valid, WholeSet= get_dataloader()


class NNPred(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,batch_size, dropout=0.2):
        super(NNPred, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bilstmhidden_size = int(hidden_size/2)
        self.num_layers = 1
        
        self.fc2fc = nn.Linear(input_size, hidden_size)
        self.fc2lstm = nn.Linear(input_size, hidden_size)
        self.fc2bilstm = nn.Linear(input_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False,batch_first=True)
        self.bilstm = nn.LSTM(hidden_size, self.bilstmhidden_size,
                              num_layers=self.num_layers,bidirectional=True,batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False,batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_size, self.bilstmhidden_size,
                              num_layers=self.num_layers,bidirectional=True,batch_first=True)

        self.dropout_fc = nn.Dropout(p=dropout)
        self.dropout_LSTM = nn.Dropout(p=dropout)
        self.dropout_BiLSTM = nn.Dropout(p=dropout)
    
        self.fc0 = nn.Linear(hidden_size,hidden_size)
        self.fc1 = nn.Linear(hidden_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        #input = tensor shape[batchsize, len, num_features]


        # init
        self.h0 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double, device=device))
        self.c0 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double, device=device))
        self.h1 = Variable(torch.randn(self.num_layers*2, self.batch_size, self.bilstmhidden_size, dtype=torch.double, device=device))
        self.c1 = Variable(torch.randn(self.num_layers*2, self.batch_size, self.bilstmhidden_size, dtype=torch.double, device=device))
        self.h2 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double, device=device))
        self.c2 = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double, device=device))
        self.h3 = Variable(torch.randn(self.num_layers*2, self.batch_size, self.bilstmhidden_size, dtype=torch.double, device=device))
        self.c3 = Variable(torch.randn(self.num_layers*2, self.batch_size, self.bilstmhidden_size, dtype=torch.double, device=device))

        in2fc = self.fc2fc(input)
        in2lstm = self.fc2lstm(input)
        in2bilstm = self.fc2bilstm(input)
        
        fc_out = self.dropout_fc(self.relu(self.fc(in2fc)))
        lstm_out,(self.h0,self.c0)= self.lstm(in2lstm.double(),( self.h0.detach(), self.c0.detach() ) )
        bilstm_out,(self.h1,self.c1) = self.bilstm(in2bilstm, ( self.h1.detach(), self.c1.detach() ) )
        lstm_out2,(self.h2,self.c2) = self.lstm2(lstm_out, ( self.h2.detach(), self.c2.detach() ) )
        bilstm_out2,(self.h3,self.c3) = self.bilstm2(bilstm_out, ( self.h3.detach(), self.c3.detach() ) )

        lstm_out2 = self.dropout_LSTM(lstm_out2)
        bilstm_out2 = self.dropout_BiLSTM(bilstm_out2)
        allsum = (lstm_out2+bilstm_out2+fc_out)
        
        out_f = self.tanh(self.fc0(allsum))
        out_f = self.tanh(self.fc1(out_f+fc_out))
        output = self.tanh(self.fc2(out_f))
        '''
        #test
        in2fc = self.relu(self.fc2fc(input))
        lstm_out,self.hiddenlstm = self.lstm(in2fc.double(),self.hiddenlstm)
        in2fc = self.tanh(self.fc0(lstm_out))
        in2fc = self.tanh(self.fc1(in2fc))
        output = self.tanh(self.fc2(in2fc))
        '''
        return output
    

def trainIters(encoder, n_iters, print_every=1000, plot_every=2, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    #criterion = nn.SmoothL1Loss()
    #criterion = nn.MSELoss(reduction='mean')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    criterion = nn.MSELoss(reduction='sum')
    count = 0
    for iter in range(1, n_iters + 1):
        for local_batch, local_labels in Training_generator:
            if local_batch.shape[0]!=32:
                #print(local_batch.shape[0])
                continue
            count = count + 1
            encoder_optimizer.zero_grad()
            local_batch = local_batch.to(device)
            #local_labels = torch.zeros(local_labels.shape,dtype=torch.double)
            local_labels = local_labels.to(device)
            predY = encoder(local_batch)
            loss = criterion(predY,local_labels).to(device)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(),0.25)
            loss.backward()
            encoder_optimizer.step()
            ls =  loss.detach().item()
            if count>=50:
                print('loss = ',ls)
                count = 0
            print_loss_total += ls
            plot_loss_total += ls
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    return plot_losses

def Eval_net(encoder):
    count = 0
    for local_batch, local_labels in Training_generator:
        if local_batch.shape[0]!=32:
            #print(local_batch.shape[0])
            continue
        count = count + 1
        local_batch = local_batch.to(device)
        local_labels = local_labels.to(device)
        predY = encoder(local_batch)
        print(WholeSet.std.repeat(32,100,1).shape)
        std = WholeSet.std.repeat(32,100,1)
        std = std[:,:,2:4].to(device)
        mn = WholeSet.mn.repeat(32,100,1)
        mn = mn[:,:,2:4].to(device)
        rg = WholeSet.range.repeat(32,100,1)
        rg = rg[:,:,2:4].to(device)
        predY = (predY*(rg*std)+mn).detach().cpu()
        pY = np.array(predY )
        local_labels = (local_labels*(rg*std)+mn).detach().cpu()
        Y = np.array(local_labels)
        for i in range(32):
            plt.figure(i)
            plt.xlim(0,80)
            plt.ylim(0,2000)
            plt.plot(pY[i,:,0],pY[i,:,1],'ro')
            plt.plot(Y[i,:,0],Y[i,:,1],'go')
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
hidden_size = 128
Prednet = NNPred(x.shape[2], y.shape[2],hidden_size, 32)

print(device)

TRAN_TAG = True
if TRAN_TAG:
    if path.exists("checkpoint.pth.tar"):
        Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    plot_losses = trainIters(Prednet, 100, print_every=4)
    torch.save(Prednet.state_dict(), 'checkpoint.pth.tar')
    showPlot(plot_losses)
else:
    Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    Prednet.eval()
    Eval_net(Prednet)
    
'''
#save
torch.save(Prednet.state_dict(), 'checkpoint.pth.tar')
#load
Prednet = NNPred(x.shape[2], y.shape[2], 32)
Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
model.eval()
'''
