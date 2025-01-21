import torch
import numpy as np
import torch.nn as nn
import scipy.io
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
dtype = torch.cuda.FloatTensor
    
class permute_change(nn.Module):
    def __init__(self, n1, n2, n3):
        super(permute_change, self).__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        
    def forward(self, x):
        x = x.permute(self.n1, self.n2, self.n3)
        return x
class TV_Loss(nn.Module):
    def __init__(self):
        super(TV_Loss, self).__init__()

    def forward(self,a):
        # gradient_a_x = torch.abs(a[:,:-1,:]-a[:,:1,:])
        # gradient_a_y = torch.abs(a[:-1,:,:]-a[1:,:,:])
        gradient_a_x = torch.abs(a[ :, :, :, :, :-1] - a[ :, :, :, :, 1:])
        gradient_a_y = torch.abs(a[ :, :, :, :-1, :] - a[ :, :, :, 1:, :])
        return gradient_a_y,gradient_a_x

class SSTV_Loss(nn.Module):
    def __init__(self):
        super(SSTV_Loss, self).__init__()

    def forward(self, a):
        # gradient_a_z = torch.abs(a[:,:,:-1]-a[:,:,:1])
        # gradient_a_yz = torch.abs(gradient_a_z[:-1,:,:] - gradient_a_z[1:,:,:])
        # gradient_a_xz = torch.abs(gradient_a_z[:,:-1,:] - gradient_a_z[:,1:,:])
        gradient_a_z = torch.abs(a[:, :, :-1, :, :] - a[:, :, 1:, :, :])
        gradient_a_yz = torch.abs(gradient_a_z[:, :, :, :-1, :] - gradient_a_z[:, :, :, 1:, :])
        gradient_a_xz = torch.abs(gradient_a_z[:, :, :, :, :-1] - gradient_a_z[:, :, :, :, 1:])
        return gradient_a_yz,gradient_a_xz
    
def prepare_image(file_name,tag = "Ohsi"):
    mat = scipy.io.loadmat(file_name)
    img_np = mat[tag]
    img_var = torch.from_numpy(img_np).type(dtype).cuda()
    return img_np,img_var
def prepare_mask(img_noisy_np):
    mask_np = np.ones(img_noisy_np.shape)
    mask_np[img_noisy_np == 0] = 0
    mask_var = torch.from_numpy(mask_np).type(dtype).cuda()
    return mask_np,mask_var

def get_show_band(data_tag):
    if data_tag == 'hsi':
        show = [0,15,30]
    if data_tag == 'rgb':
        show = [0,1,2]
    if data_tag == 'video':
        show = [27,27,27]
    return show
def evaluation_matrix(data_tag, x1_np, x2_np):
    psnr_here = PSNR(x1_np, x2_np, data_range = 1)
    if data_tag == 'rgb':
        ssim_here = SSIM(x1_np, x2_np, channel_axis = 2, data_range = 1)
    else:
        ssim_here = SSIM(x1_np, x2_np, data_range = 1)
    return psnr_here,ssim_here

