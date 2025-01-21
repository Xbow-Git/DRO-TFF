import torch
from torch import nn, optim 
from torch.autograd import Variable 
import os
import numpy as np
import math
dtype = torch.cuda.FloatTensor

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)    
            
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
        
    def create_depth_layer(r):
        return nn.Sequential(nn.Linear(r, r, bias = False)
                                 ,nn.LeakyReLU())
def create_depth_layer(r):
        return nn.Sequential(nn.Linear(r, r, bias = False)
                                 ,nn.LeakyReLU())
class permute_change(nn.Module):
    def __init__(self, n1, n2, n3):
        super(permute_change, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
    def forward(self, x):
        x = x.permute(self.n1, self.n2, self.n3)
        
class DRO_TFF(nn.Module): 
    def __init__(self,n_1,n_2,n_3,r,omega,num_deep_layer):
        super(DRO_TFF, self).__init__()
        self.A_mid_channel = 300
        self.B_mid_channel = 300
        self.H_mid_channel = 100
        self.n_3 = n_3
        self.A_net = nn.Sequential(SineLayer(1, self.A_mid_channel,omega),
                                        nn.Linear(self.A_mid_channel, r))
        self.A_input = torch.from_numpy(
                np.array(range(1,n_1+1))).reshape(n_1, 1).type(dtype)
            
        self.B_net = nn.Sequential(SineLayer(1, self.B_mid_channel, omega),
                                        nn.Linear(self.B_mid_channel, r))
        self.B_input = torch.from_numpy(
                np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
        self.net = nn.Sequential(*[create_depth_layer(r) 
                                       for _ in range(num_deep_layer) ])
            
        self.H_net = nn.Sequential(SineLayer(1, self.H_mid_channel,omega),
                                       nn.Linear(self.H_mid_channel, r))
        self.H_input = torch.from_numpy(
                np.array(range(1,n_3+1))).reshape(n_3,1).type(dtype)
        
    def forward(self):
        self.A_hat = self.A_net(self.A_input).permute(1,0).unsqueeze(-1) # n3*expand, n1, 1
        self.B_hat = self.B_net(self.B_input).permute(1,0).unsqueeze(-2) # n3*expand, 1, n2
            
        x = torch.matmul(self.A_hat,self.B_hat)
        H = self.H_net(self.H_input).permute(1,0)
        return self.net(x.permute(1,2,0)) @ H
        
class DRO_TFF_video(nn.Module): 
    def __init__(self,n_1,n_2,n_3,r,omega,num_deep_layer):
        super(DRO_TFF_video, self).__init__()
        self.A_mid_channel = 300
        self.B_mid_channel = 300
        self.n_3 = n_3
        self.A_net = nn.Sequential(SineLayer(1, self.A_mid_channel,omega),
                                        nn.Linear(self.A_mid_channel, r))
        self.A_input = torch.from_numpy(
                np.array(range(1,n_1+1))).reshape(n_1,1).type(dtype)
            
        self.B_net = nn.Sequential(SineLayer(1, self.B_mid_channel,omega),
                                        nn.Linear(self.B_mid_channel, r))
        self.B_input = torch.from_numpy(
                np.array(range(1,n_2+1))).reshape(n_2,1).type(dtype)
        self.net = nn.Sequential(nn.Linear(r,r,bias = False),
                                 nn.LeakyReLU(),
                                 nn.Linear(r,n_3,bias = False))
        
    def forward(self):
        self.A_hat = self.A_net(self.A_input).permute(1,0).unsqueeze(-1) # r, n1, 1
        self.B_hat = self.B_net(self.B_input).permute(1,0).unsqueeze(-2) # r, 1, n2
            
        x = torch.matmul(self.A_hat,self.B_hat)
        return self.net(x.permute(1,2,0))