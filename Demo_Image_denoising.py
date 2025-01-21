import torch
from torch import nn, optim 
from torch.autograd import Variable 
import os
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from utils import *
from model.drotff import *
import scipy.io as scio
import matplotlib.pyplot as plt 
import scipy.io
import math
import random
dtype = torch.cuda.FloatTensor

def main(gamma = 4e-05, max_iter = 4000, lr_real = 0.002, omega = 2, 
         num_deep_layer = 3, beta = 0.7):
    
    for data in data_all:
        for c in c_all:
            print(data, c, "Loading...")
            seed=1
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            
            file_name = f'./data/{data}gt.mat'
            mat = scipy.io.loadmat(file_name)
            gt_np = mat["Ohsi"]
            gt = torch.from_numpy(gt_np).type(dtype).cuda()
            
            file_name = f'./data/{data}{c}.mat'
            mat = scipy.io.loadmat(file_name)
            X_np = mat["Nhsi"]
            X = torch.from_numpy(X_np).type(dtype).cuda()
            n3_real = X.shape[2]
            
            r = int(X.shape[0]*beta)
            
            model = DRO_TFF(X.shape[0],X.shape[1],X.shape[2],
                            r,omega,num_deep_layer).type(dtype)
            
            params = []
            params += [x for x in model.parameters()]
            
            s = sum([np.prod(list(p.size())) for p in params]); 
            print('Number of params: %d' % s)
            optimizier = optim.Adam(params, lr=lr_real, weight_decay=10e-8) 
            
            psnr_best = 0
            import time
            start_time = time.time()
            for iter in range(max_iter):
                X_Out = model()
                loss = torch.norm(X_Out-X,2)
                
                # TV and SSTV loss
                loss = loss + gamma*torch.norm(X_Out[1:,:,:]-X_Out[:-1,:,:], 1)
                loss = loss + gamma*torch.norm(X_Out[:,1:,:]-X_Out[:,:-1,:], 1)
                dz = X_Out[:,:,1:] - X_Out[:,:,:-1]
                loss = loss + 10*gamma*torch.norm(dz[1:,:,1:]-dz[:-1,:,1:], 1)
                loss = loss + 10*gamma*torch.norm(dz[:,1:,1:]-dz[:,:-1,1:], 1)
                
                optimizier.zero_grad()
                loss.backward(retain_graph=True)
                optimizier.step()
                
                X_Out_np = X_Out.cpu().detach().numpy()
                
                if iter % 100 == 0:
                    psnr_here,ssim_here = evaluation_matrix(data_tag, gt_np, X_Out_np)
                    if psnr_here > psnr_best:
                        psnr_best = psnr_here
                        ssim_best = ssim_here
                        X_save = X_Out_np
                    print('iter:{},psnr_here:{:.3f}'.format(iter,psnr_here))
                    if image_show:
                        plt.figure(figsize=(11,33))
                        plt.subplot(131)
                        plt.imshow(np.clip(np.stack((gt_np[:,:,show[0]],
                                         gt_np[:,:,show[1]],
                                         gt_np[:,:,show[2]]),2),0,1))
                        plt.title('Observed')
                    
                        plt.subplot(132)
                        plt.imshow(np.clip(np.stack((X_np[:,:,show[0]],
                                         X_np[:,:,show[1]],
                                         X_np[:,:,show[2]]),2),0,1))
                        plt.title('incomplete')
            
                        plt.subplot(133)
                        plt.imshow(np.clip(np.stack((X_Out_np[:,:,show[0]],
                                         X_Out_np[:,:,show[1]],
                                         X_Out_np[:,:,show[2]]),2),0,1))
                        plt.title('Recovered')
                        plt.show()
                        
            end_time = time.time()
            print('time cost:{:.1f},psnr_best:{:.3f},ssim_best:{:.3f}'.format(end_time-start_time,psnr_best,ssim_best))
            
            if mat_save:
                save_folder = f'./result/{data}/'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                scio.savemat(os.path.join(save_folder,
                                      f"{c}_DRO_TFF.mat"), {'x_recon':X_save, 'psnr':psnr_best,'ssim':ssim_best})
            
            
############# Data set #############
data_tag = 'hsi';data_all = ['fake_and_real_peppers'];c_all = ["0.2"]
show = [21,15,7]#get_show_band(data_tag)

#############################################参数设定############################################
gamma = 0.00004
max_iter = 4000
lr_real = 0.002
beta = 0.3  
num_deep_layer = 3
gamma = 4e-05

# omega is used for INRs
# set it to 1 or 2 for a better result
omega = 1

# For getting Time, set image_show, mat_save to False, False
image_show = False
mat_save = False

main(gamma, max_iter, lr_real, omega, num_deep_layer, beta)