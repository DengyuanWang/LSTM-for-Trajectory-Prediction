# -*- coding: utf-8 -*-
from io import open
import os.path
from os import path
import random
import numpy as np

import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
import matplotlib.ticker as ticker
import numpy as np
BatchSize = 128
class TrajectoryDataset(Dataset):
    """Face Landmarks dataset."""
    
    def __init__(self, csv_file='./data/WholeVdata2.csv'):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """
        self.csv_file = csv_file;
        # store X as a list, each element is a 100*42(len * attributes num) np array [velx;vely;x;y;acc;angle] * 7
        self.X_frames = []
        # store Y as a list, each element is a 100*4(len * attributes num) np array[velx;vely;x;y]
        self.Y_frames = []
        self.load_data()
        self.normalize_data()
    
    def __len__(self):
        return len(self.X_frames)
    
    def __getitem__(self, idx):
        single_data = self.X_frames[idx]
        single_label = self.Y_frames[idx]
        return (single_data, single_label)
    
    def load_data(self):
        dataS = pd.read_csv(self.csv_file)
        max_vehiclenum = np.max(dataS.Vehicle_ID.unique())
        for vid in dataS.Vehicle_ID.unique():
            print('{0} and {1}'.format(vid, max_vehiclenum))
            frame_ori = dataS[dataS.Vehicle_ID == vid]
            frame = frame_ori[['Local_X', 'Local_Y', 'v_Acc', 'Angle',
                               'L_rX', 'L_rY', 'L_rAcc', 'L_angle',
                               'F_rX', 'F_rY', 'F_rAcc', 'F_angle',
                               'LL_rX', 'LL_rY', 'LL_rAcc', 'LL_angle',
                               'LF_rX', 'LF_rY', 'LF_rAcc', 'LF_angle',
                               'RL_rX', 'RL_rY', 'RL_rAcc', 'RL_angle',
                               'RF_rX', 'RF_rY', 'RF_rAcc', 'RF_angle']]
            frame = np.asarray(frame)
            frame[np.where(frame>4000)] = 0 # assign all 5000 to 0
            # remove anomalies, which has a discontinuious local x or local y
            dis = frame[1:,:2] - frame[:-1,:2]
            dis = np.sqrt(np.power(dis[:,0],2)+np.power(dis[:,1],2))
            idx = np.where(dis>10)
            if not (idx[0].all):
                continue
            #smooth the data column wise
            #window size = 5, polynomial order = 3
            frame =  scipy.signal.savgol_filter(frame, window_length=5, polyorder=3,axis=0)
            #print(frame[:,0])
            # calculate vel_x and vel_y according to local_x and local_y for all vehi
            All_vels = []
            for i in range(7):
                '''
                plt.figure(1)
                plt.subplot(2,3,1)
                print(0+i*4)
                plt.plot(frame[1:,(0+i*4)],'ro')
                plt.subplot(2,3,2)
                plt.plot(frame[1:,1+i*4],'ro')
                '''
                x_vel = (frame[1:,0+i*4] -  frame[:-1, 0+i*4])/0.1;
                v_avg = (x_vel[1:] +x_vel[:-1])/2.0;
                v1 = [2.0*x_vel[0]- v_avg[0]];v_end = [2.0*x_vel[-1]- v_avg[-1]];
                vel = (v1+ v_avg.tolist()+v_end)
                vel = np.array(vel)
                
                y_vel = (frame[1:,1+i*4] -  frame[:-1, 1+i*4])/0.1;
                vy_avg = (y_vel[1:] +y_vel[:-1])/2.0;
                vy1 = [2.0*y_vel[0]- vy_avg[0]];vy_end = [2.0*y_vel[-1]- vy_avg[-1]];
                vely = (vy1+ vy_avg.tolist()+ vy_end)
                vely = np.array(vely)
                '''
                plt.subplot(2,3,4)
                plt.plot(vel,'ro')
                plt.subplot(2,3,5)
                plt.plot(vely,'ro')
                plt.subplot(2,3,6)
                print(frame[1:50,0+i*4])
                plt.plot(frame[1:,0+i*4],frame[1:,1+i*4],'go')
                plt.show()
                '''
                if isinstance(All_vels,(list)):
                    All_vels = np.vstack((vel,vely))
                else:
                    All_vels = np.vstack((All_vels,vel.reshape(1,-1)))
                    All_vels = np.vstack((All_vels,vely.reshape(1,-1)))
            All_vels = np.transpose(All_vels)
            total_frame_data = np.concatenate(( All_vels[:,:2], frame),axis=1)
            #total_frame_data = np.concatenate(( All_vels[:,:2], frame, All_vels[:,2:]),axis=1)
            # split into several frames each frame have a total length of 100, drop sequence smaller than 130
            if(total_frame_data.shape[0]<130):
                continue
            
            X = total_frame_data[:-29,:]
            Y = total_frame_data[29:,:4]
            
            
            
            count = 0
            for i in range(X.shape[0]-100):
                if random.random()>0.2:
                    continue
                j = i-1;
                if count>20:
                    break
                #print('X[] shape',X[i:i+100,:].shape)
                self.X_frames = self.X_frames + [X[i:i+100,:]]
                self.Y_frames = self.Y_frames + [Y[i:i+100,:]]
                count = count+1
    def normalize_data(self):
        A = [list(x) for x in zip(*(self.X_frames))]
        A = torch.tensor(A)
        A = A.view(-1,A.shape[2])
        print('A:',A.shape)
        self.mn = torch.mean(A,dim=0)
        self.range = (torch.max(A,dim=0).values-torch.min(A,dim=0).values)/2.0
        self.range = torch.ones(self.range.shape,dtype = torch.double)
        self.std = torch.std(A,dim=0)
        #self.X_frames = [torch.tensor(item) for item in self.X_frames]
        #self.Y_frames = [torch.tensor(item) for item in self.Y_frames]
        self.X_frames = [(torch.tensor(item)-self.mn)/(self.std*self.range) for item in self.X_frames]
        self.Y_frames = [(torch.tensor(item)-self.mn[:4])/(self.std[:4]*self.range[:4]) for item in self.Y_frames]

def get_dataloader():
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    #load dataset
    if path.exists("my_dataset.pickle"):
        with open('my_dataset.pickle', 'rb') as data:
            dataset = pickle.load(data)
    else:
        dataset = TrajectoryDataset()
        with open('my_dataset.pickle', 'wb') as output:
            pickle.dump(dataset, output)
    #split dataset into train test and validation 7:2:1
    num_train = (int)(dataset.__len__()*0.7)
    num_test = (int)(dataset.__len__()*0.9) - num_train
    num_validation = (int)(dataset.__len__()-num_test-num_train)
    train, test, validation = torch.utils.data.random_split(dataset, [num_train, num_test, num_validation])
    train_loader = DataLoader(train, batch_size=BatchSize, shuffle=True)
    test_loader = DataLoader(test, batch_size=BatchSize, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=BatchSize, shuffle=True)
    return (train_loader, test_loader, validation_loader, dataset)

