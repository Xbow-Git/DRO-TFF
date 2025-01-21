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
import time
import random
dtype = torch.cuda.FloatTensor

def main(gamma = 4e-05, max_iter = 4000, lr_real = 0.002, omega = 2, 
         num_deep_layer = 3, beta = 0.7):
    
    TV = TV_Loss()
    SSTV = SSTV_Loss()
    class soft(nn.Module):
        def __init__(self):
            super(soft, self).__init__()
        
        def forward(self, x, lam):
            x_abs = x.abs()-lam
            zeros = x_abs - x_abs
            n_sub = torch.max(x_abs, zeros)
            x_out = torch.mul(torch.sign(x), n_sub)
            return x_out
    
    def closure(iter,mask):
        global ssim_best,psnr_best,X_save
        
        if iter == 0:
            ps_best = 0
            ssim_best = 0

        out_ = model().permute(2,0,1)
        out_ = out_[None,None, :]
        mask = mask.permute(2,0,1)
        
        D_x_,D_y_ = TV(out_)
        D_xz_, D_yz_ = SSTV(out_)
        D_x = D_x_.clone().detach()
        D_y = D_y_.clone().detach()
        D_xz = D_xz_.clone().detach()
        D_yz = D_yz_.clone().detach()
        out = out_.clone().detach()

        img_noisy_var = X.permute(2,0,1)[None,None,:]
        img_noisy_np = img_noisy_var.detach().cpu().numpy()
        if iter == 0:
            global D_1,D_2,D_3,D_4,D_5,V_1,V_2,V_3,V_4,V_5,S,mu,thres,thres_tv,thres_sstv
            psnr_best = 0
            D_2 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2],
                            img_noisy_var.shape[3]-1,img_noisy_var.shape[4]]).type(dtype)
            D_3 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2],
                            img_noisy_var.shape[3],img_noisy_var.shape[4]-1]).type(dtype)
            D_4 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2]-1,
                            img_noisy_var.shape[3]-1,img_noisy_var.shape[4]]).type(dtype)
            D_5 = torch.zeros([img_noisy_var.shape[0],img_noisy_var.shape[1],img_noisy_var.shape[2]-1,
                            img_noisy_var.shape[3],img_noisy_var.shape[4]-1]).type(dtype)

            V_2 = D_x.type(dtype)
            V_3 = D_y.type(dtype)
            V_4 = D_xz.type(dtype)
            V_5 = D_yz.type(dtype)

            S = (img_noisy_var-out).type(dtype)

        S = soft_thres(img_noisy_var-out, thres)

        V_2 = soft_thres(D_x + D_2 / mu, thres_tv)
        V_3 = soft_thres(D_y + D_3 / mu, thres_tv)

        V_4 = soft_thres(D_xz + D_4 / mu,thres_sstv)
        V_5 = soft_thres(D_yz + D_5 / mu,thres_sstv)

        total_loss = mu/2 * torch.norm(D_x_-(V_2-D_2/mu),2)
        total_loss += mu/2 * torch.norm(D_y_-(V_3-D_3/mu),2)
        total_loss += 10*mu/2 * torch.norm(D_xz_-(V_4-D_4/mu),2)
        total_loss += 10*mu/2 * torch.norm(D_yz_-(V_5-D_5/mu),2)
        total_loss += torch.norm(img_noisy_var*mask-out_*mask-S,2)

        total_loss.backward()
        D_2 = (D_2 + mu * (D_x  - V_2)).clone().detach()
        D_3 = (D_3 + mu * (D_y  - V_3)).clone().detach()
        D_4 = (D_4 + mu * (D_xz  - V_4)).clone().detach()
        D_5 = (D_5 + mu * (D_yz  - V_5)).clone().detach()

        
        out_np = out.detach().squeeze().permute(1,2,0).cpu().numpy()
        img_noisy_np = img_noisy_np.squeeze()
        img_np = gt_np.transpose(2,0,1)
        
        if iter % 100 == 0:
            psnr_here,ssim_here = evaluation_matrix(data_tag, gt_np, out_np)
            print('iter:{},psnr_here:{:.3f}'.format(iter,psnr_here))
            if psnr_here > psnr_best:
                psnr_best = psnr_here
                ssim_best = ssim_here
                X_save = out_np
                
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
                plt.imshow(np.clip(np.stack((out_np[:,:,show[0]],
                                         out_np[:,:,show[1]],
                                         out_np[:,:,show[2]]),2),0,1))
                plt.title('Recovered')
                plt.show()

        return psnr_best,ssim_best,total_loss

    for data in data_all:
        for c in c_all:
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
            
            mask = torch.ones(X.shape).type(dtype)
            mask[X == 0] = 0 
            X[mask == 0] = 0
            
            params = []
            params += [x for x in model.parameters()]
            
            s = sum([np.prod(list(p.size())) for p in params]); 
            print('Number of params: %d' % s)
            optimizier = optim.Adam(params, lr=lr_real, weight_decay=10e-8) 
        
            show = [25,14,5]
            ps_best = 0
            ssim_best = 0
            soft_thres=soft()
            
            start_time = time.time()
            for iter in range(max_iter):
                optimizier.zero_grad()
                ps_best, ssim_best, total_loss = closure(iter, mask)
                optimizier.step()
                
            end_time = time.time()
            print('time cost:{:.1f},psnr_best:{:.3f},ssim_best:{:.3f}'.format(end_time-start_time,psnr_best,ssim_best))
            if mat_save:
                save_folder = f'./result/{data}/'
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                scio.savemat(os.path.join(save_folder,
                f"{c}_DRO_TFF.mat"), {'x_recon':X_save, 'psnr':ps_best,'ssim':ssim_best, 'params':s})


############# Data set #############
data_tag = 'hsi'
data_all = ['cloth']
c_all = ["c3"]
show = get_show_band(data_tag)

#############################################参数设定############################################

# HSI from CAVE
gamma = 4e-05
max_iter = 4000
lr_real = 0.002
beta = 0.3
num_deep_layer = 3
mu = 0.04
alpha_3 = 0.01
thres = 2 * alpha_3
thres_tv = 0.001
thres_sstv = 0.01
# omega is used for INRs
# set it to 1 or 2 for a better result
omega = 2

# For getting time consumption, 
# set image_show, mat_save = False, False 
image_show = False
mat_save = False

main(gamma, max_iter, lr_real, omega, num_deep_layer, beta)