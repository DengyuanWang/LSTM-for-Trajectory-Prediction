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

from dataPrepare import *

SOS_token = [0.0,0.0,0.0,0.0]
EOS_token = [1.0,1.0,1.0,1.0]
MAX_LENGTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Training_generator, Test, Valid = get_dataloader()


class NNPred(nn.Module):
    def __init__(self, input_size, output_size,batch_size, dropout=0.2):
        super(NNPred, self).__init__()
        
        self.batch_size = batch_size
        self.hidden_size = 256
        self.bilstmhidden_size = 128
        self.num_layers = 3
        self.reset()
        
        self.fc2fc = nn.Linear(input_size, hidden_size)
        self.fc2lstm = nn.Linear(input_size, hidden_size)
        self.fc2bilstm = nn.Linear(input_size, hidden_size)
        
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False)
        self.bilstm = nn.LSTM(hidden_size, self.bilstmhidden_size,
                              num_layers=self.num_layers,bidirectional=True)
        self.dropout_fc = nn.Dropout(p=dropout)
        self.dropout_LSTM = nn.Dropout(p=dropout)
        self.dropout_BiLSTM = nn.Dropout(p=dropout)
    
        self.fc0 = nn.Linear(hidden_size,hidden_size)
        self.fc1 = nn.Linear(hidden_size,hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2,output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        #input = tensor shape[len,batchsize,features]
        in2fc = self.relu(self.fc2fc(input))
        in2lstm = self.relu(self.fc2lstm(input))
        in2bilstm = self.relu(self.fc2bilstm(input))
        fc_out = self.dropout_fc(self.relu(self.fc(in2fc)))
        lstm_out,self.hiddenlstm = self.lstm(in2lstm.double(),self.hiddenlstm)
        bilstm_out,self.hiddenbilstm = self.bilstm(in2bilstm,self.hiddenbilstm)
        lstm_out = self.dropout_LSTM(lstm_out)
        bilstm_out = self.dropout_BiLSTM(bilstm_out)
        lstm_out = torch.sigmoid(lstm_out)
        allsum = lstm_out*fc_out+(1-lstm_out)*bilstm_out
        #allsum = fc_out + fc_out*(lstm_out+bilstm_out)
        
        out_f = self.tanh(self.fc0(allsum))
        out_f = self.tanh(self.fc1(out_f))
        output = self.tanh(self.fc2(out_f))
        return output
    
    def reset(self):
        self.hiddenlstm = (torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double, device=device),torch.zeros(self.num_layers, self.batch_size, self.hidden_size, dtype=torch.double, device=device))
        self.hiddenbilstm = (torch.zeros(self.num_layers*2, self.batch_size, self.bilstmhidden_size, dtype=torch.double, device=device),torch.zeros(self.num_layers*2, self.batch_size, self.bilstmhidden_size, dtype=torch.double, device=device))


def trainIters(encoder, n_iters, print_every=1000, plot_every=2, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    #criterion = nn.SmoothL1Loss()
    criterion = nn.MSELoss(reduction='mean')
    Xsample = []
    Ysample = []
    for iter in range(1, n_iters + 1):
        for local_batch, local_labels in Training_generator:
            if isinstance(Xsample,(list)):
                Xsample = local_batch
                Ysample = local_labels
            if Xsample.shape != local_batch.shape or Ysample.shape !=local_labels.shape:
                continue
            encoder.zero_grad()
            encoder_optimizer.zero_grad()
            encoder.reset()
            local_batch = torch.transpose(local_batch, 0, 1)
            local_labels =  torch.transpose(local_labels, 0, 1)
            predY = encoder(local_batch)
            loss = criterion(predY,local_labels)
            loss.backward()
            encoder_optimizer.step()
            print(loss.item())
            print_loss_total += loss
            plot_loss_total += loss
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                                iter, iter / n_iters * 100, print_loss_avg))
            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0
    showPlot(plot_losses)


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

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

train_iter = iter(Training_generator)
x, y = train_iter.next()

hidden_size = 256
Prednet = NNPred(x.shape[2], y.shape[2], 32)
Prednet.double()
Prednet.to(device)
trainIters(Prednet, 60, print_every=5)
torch.save(Prednet.state_dict(), 'checkpoint.pth.tar')
'''
#save
torch.save(Prednet.state_dict(), 'checkpoint.pth.tar')
#load
Prednet = NNPred(x.shape[2], y.shape[2], 32)
Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
model.eval()
'''
