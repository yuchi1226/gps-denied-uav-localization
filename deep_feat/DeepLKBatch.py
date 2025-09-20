import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import requests
from PIL import Image
from torch.nn.functional import grid_sample
from pdb import set_trace as st
from sys import argv
import argparse
import time
from math import cos, sin, pi, sqrt
import sys
import time
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 添加安全全局變量
try:
    from torch.serialization import add_safe_globals
    # 添加 torchvision 模型到安全列表
    import torchvision.models as models
    add_safe_globals([models.vgg.VGG])
except ImportError:
    # 如果 PyTorch 版本較低，忽略此操作
    pass

class InverseBatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        batch_size, h, w = input.size()
        assert(h == w)
        H = torch.zeros(batch_size, h, h, device=input.device, dtype=input.dtype)
        for i in range(0, batch_size):
            H[i, :, :] = torch.inverse(input[i, :, :])
        ctx.save_for_backward(H)
        return H

    @staticmethod
    def backward(ctx, grad_output):
        H, = ctx.saved_tensors
        batch_size, h, w = H.size()
        assert(h == w)
        
        Hl = H.transpose(1,2).repeat(1, 1, h).view(batch_size*h*h, h, 1)
        Hr = H.repeat(1, h, 1).view(batch_size*h*h, 1, h)
        
        r = Hl.bmm(Hr).view(batch_size, h, h, h, h) * \
            grad_output.contiguous().view(batch_size, 1, 1, h, h).expand(batch_size, h, h, h, h)
        return -r.sum(-1).sum(-1)

def InverseBatchFun(input):
    batch_size, h, w = input.size()
    assert(h == w)
    H = torch.zeros(batch_size, h, h, device=input.device, dtype=input.dtype)
    for i in range(0, batch_size):
        H[i, :, :] = torch.inverse(input[i, :, :])
    return H

class GradientBatch(nn.Module):
    def __init__(self):
        super(GradientBatch, self).__init__()
        wx = torch.FloatTensor([-.5, 0, .5]).view(1, 1, 1, 3)
        wy = torch.FloatTensor([[-.5], [0], [.5]]).view(1, 1, 3, 1)
        self.register_buffer('wx', wx)
        self.register_buffer('wy', wy)
        self.padx_func = torch.nn.ReplicationPad2d((1,1,0,0))
        self.pady_func = torch.nn.ReplicationPad2d((0,0,1,1))

    def forward(self, img):
        batch_size, k, h, w = img.size()
        img_ = img.view(batch_size * k, 1, h, w)
        
        img_padx = self.padx_func(img_)
        img_dx = torch.nn.functional.conv2d(
            input=img_padx,
            weight=self.wx,
            stride=1,
            padding=0
        ).squeeze(1)
        
        img_pady = self.pady_func(img_)
        img_dy = torch.nn.functional.conv2d(
            input=img_pady,
            weight=self.wy,
            stride=1,
            padding=0
        ).squeeze(1)
        
        img_dx = img_dx.view(batch_size, k, h, w)
        img_dy = img_dy.view(batch_size, k, h, w)
        
        return img_dx, img_dy

def normalize_img_batch(img):
    # per-channel zero-mean and unit-variance of image batch
    N, C, H, W = img.size()
    
    # compute per channel mean for batch, subtract from image
    mean = img.mean(dim=(2, 3), keepdim=True)
    img_ = img - mean
    
    # compute per channel std dev for batch, divide img
    std_dev = img.std(dim=(2, 3), keepdim=True)
    img_ = img_ / std_dev
    
    return img_

def warp_hmg(img, p):
    batch_size, k, h, w = img.size()
    
    x = torch.arange(w, device=img.device)
    y = torch.arange(h, device=img.device)
    
    X, Y = meshgrid(x, y)
    
    # create xy matrix, 2 x N
    xy = torch.stack([X.reshape(-1), Y.reshape(-1), torch.ones(X.numel(), device=img.device)], dim=0)
    xy = xy.repeat(batch_size, 1, 1)
    
    H = param_to_H(p)
    xy_warp = H.bmm(xy)
    
    # extract warped X and Y, normalizing the homog coordinates
    X_warp = xy_warp[:,0,:] / xy_warp[:,2,:]
    Y_warp = xy_warp[:,1,:] / xy_warp[:,2,:]
    
    X_warp = X_warp.view(batch_size, h, w) + (w-1)/2
    Y_warp = Y_warp.view(batch_size, h, w) + (h-1)/2
    
    img_warp, mask = grid_bilinear_sampling(img, X_warp, Y_warp)
    return img_warp, mask

def grid_bilinear_sampling(A, x, y):
    batch_size, k, h, w = A.size()
    x_norm = x/((w-1)/2) - 1
    y_norm = y/((h-1)/2) - 1
    grid = torch.stack([x_norm, y_norm], dim=-1).view(batch_size, h, w, 2)
    Q = grid_sample(A, grid, mode='bilinear', align_corners=False)
    
    in_view_mask = ((x_norm > -1+2/w) & (x_norm < 1-2/w) & 
                   (y_norm > -1+2/h) & (y_norm < 1-2/h))
    
    return Q, in_view_mask

def param_to_H(p):
    batch_size, _, _ = p.size()
    z = torch.zeros(batch_size, 1, 1, device=p.device, dtype=p.dtype)
    p_ = torch.cat((p, z), 1)
    I = torch.eye(3, 3, device=p.device, dtype=p.dtype).repeat(batch_size, 1, 1)
    H = p_.view(batch_size, 3, 3) + I
    return H

def H_to_param(H):
    batch_size, _, _ = H.size()
    I = torch.eye(3, 3, device=H.device, dtype=H.dtype).repeat(batch_size, 1, 1)
    p = H - I
    p = p.view(batch_size, 9, 1)
    p = p[:, 0:8, :]
    return p

def meshgrid(x, y):
    imW = x.size(0)
    imH = y.size(0)
    
    x = x - x.max()/2
    y = y - y.max()/2
    
    X = x.unsqueeze(0).repeat(imH, 1)
    Y = y.unsqueeze(1).repeat(1, imW)
    return X, Y

class vgg16Conv(nn.Module):
    def __init__(self, model_path):
        super(vgg16Conv, self).__init__()
        
        print('Loading pretrained network...', end='')
        try:
            vgg16 = torch.load(model_path, map_location=device, weights_only=True)
        except:
            vgg16 = torch.load(model_path, map_location=device,weights_only=False)
        print('done')
        
        self.features = nn.Sequential(*(list(vgg16.features.children())[0:15]))
        
        # freeze conv1, conv2
        for p in self.parameters():
            if p.size()[0] < 256:
                p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        return x

class noPoolNet(nn.Module):
    def __init__(self, model_path):
        super(noPoolNet, self).__init__()
        
        print('Loading pretrained network...', end='')
        vgg16 = torch.load(model_path, map_location=device, weights_only=False)
        print('done')
        
        vgg_features = list(vgg16.features.children())
        vgg_features[2].stride = (2,2)
        vgg_features[7].stride = (2,2)
        
        self.custom = nn.Sequential(*(vgg_features[0:4] + 
                                     vgg_features[5:9] + 
                                     vgg_features[10:15]))
        
        layer = 0
        for p in self.parameters():
            if layer < 8:
                p.requires_grad = False
            layer += 1

    def forward(self, x):
        x = self.custom(x)
        return x

class vgg16fineTuneAll(nn.Module):
    def __init__(self, model_path):
        super(vgg16fineTuneAll, self).__init__()
        
        print('Loading pretrained network...', end='')
        vgg16 = torch.load(model_path, map_location=device, weights_only=False)
        print('done')
        
        self.features = nn.Sequential(*(list(vgg16.features.children())[0:15]))

    def forward(self, x):
        x = self.features(x)
        return x

class custom_net(nn.Module):
    def __init__(self, model_path):
        super(custom_net, self).__init__()
        
        print('Loading pretrained network...', end='')
        self.custom = torch.load(model_path, map_location=device, weights_only=False)
        print('done')

    def forward(self, x):
        x = self.custom(x)
        return x

class custConv(nn.Module):
    def __init__(self, model_path):
        super(custConv, self).__init__()
        
        print('Loading pretrained network...', end='')
        self.custom = torch.load(model_path, map_location=device)
        print('done')

    def forward(self, x):
        x = self.custom(x)
        return x

class DeepLK(nn.Module):
    def __init__(self, conv_net):
        super(DeepLK, self).__init__()
        self.img_gradient_func = GradientBatch()
        self.conv_func = conv_net
        self.inv_func = InverseBatch.apply

    def forward(self, img, temp, init_param=None, tol=1e-3, max_itr=500, conv_flag=0, ret_itr=False):
        if conv_flag:
            Ft = self.conv_func(temp)
            Fi = self.conv_func(img)
        else:
            Fi = img
            Ft = temp
            
        batch_size, k, h, w = Ft.size()
        
        Ftgrad_x, Ftgrad_y = self.img_gradient_func(Ft)
        dIdp = self.compute_dIdp(Ftgrad_x, Ftgrad_y)
        dIdp_t = dIdp.transpose(1, 2)
        
        invH = self.inv_func(dIdp_t.bmm(dIdp))
        invH_dIdp = invH.bmm(dIdp_t)
        
        if init_param is None:
            p = torch.zeros(batch_size, 8, 1, device=img.device)
        else:
            p = init_param
            
        dp = torch.ones(batch_size, 8, 1, device=img.device)  # ones so norm > tol for first iteration
        
        itr = 1
        
        while (dp.norm(p=2, dim=1, keepdim=True).max() > tol or itr == 1) and (itr <= max_itr):
            Fi_warp, mask = warp_hmg(Fi, p)
            mask = mask.unsqueeze(1).repeat(1, k, 1, 1)
            
            Ft_mask = Ft * mask
            r = Fi_warp - Ft_mask
            r = r.view(batch_size, k * h * w, 1)
            
            dp_new = invH_dIdp.bmm(r)
            dp_new[:,6:8,0] = 0
            
            # Only update if norm is above tolerance
            dp_norm_condition = (dp.norm(p=2, dim=1, keepdim=True) > tol).float()
            dp = dp_norm_condition * dp_new
            
            p = p - dp
            itr = itr + 1
            
        print('finished at iteration ', itr)
        
        if ret_itr:
            return p, param_to_H(p), itr
        else:
            return p, param_to_H(p)

    def compute_dIdp(self, Ftgrad_x, Ftgrad_y):
        batch_size, k, h, w = Ftgrad_x.size()
        
        x = torch.arange(w, device=Ftgrad_x.device)
        y = torch.arange(h, device=Ftgrad_x.device)
        
        X, Y = meshgrid(x, y)
        X = X.view(-1, 1).repeat(batch_size, k, 1)
        Y = Y.view(-1, 1).repeat(batch_size, k, 1)
        
        Ftgrad_x = Ftgrad_x.view(batch_size, k * h * w, 1)
        Ftgrad_y = Ftgrad_y.view(batch_size, k * h * w, 1)
        
        dIdp = torch.cat((
            X * Ftgrad_x, 
            Y * Ftgrad_x,
            Ftgrad_x,
            X * Ftgrad_y,
            Y * Ftgrad_y,
            Ftgrad_y,
            -X * X * Ftgrad_x - X * Y * Ftgrad_y,
            -X * Y * Ftgrad_x - Y * Y * Ftgrad_y), dim=2)
        
        return dIdp

def main():
    if len(argv) < 3:
        print("Usage: python DeepLKBatch.py image1_path image2_path")
        return
        
    sz = 200
    xy = [0, 0]
    sm_factor = 8
    sz_sm = int(sz/sm_factor)
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    img1 = Image.open(argv[1]).crop((xy[0], xy[1], xy[0]+sz, xy[1]+sz))
    img1_coarse = preprocess(img1.resize((sz_sm, sz_sm))).to(device)
    img1 = preprocess(img1).to(device)
    
    img2 = Image.open(argv[2]).crop((xy[0], xy[1], xy[0]+sz, xy[1]+sz))
    img2_coarse = preprocess(img2.resize((sz_sm, sz_sm))).to(device)
    img2 = preprocess(img2).to(device)
    
    scale = 1.6
    angle = 15
    projective_x = 0
    projective_y = 0
    translation_x = 0
    translation_y = 0
    
    rad_ang = angle / 180 * pi
    
    p = torch.Tensor([scale + cos(rad_ang) - 2,
                     -sin(rad_ang),
                     translation_x,
                     sin(rad_ang),
                     scale + cos(rad_ang) - 2,
                     translation_y,
                     projective_x, 
                     projective_y]).to(device)
    p = p.view(8, 1)
    pt = p.repeat(5, 1, 1)
    
    dlk = DeepLK(None)  # Need to provide a conv_net for real usage
    
    img1 = img1.repeat(5, 1, 1, 1)
    img2 = img2.repeat(5, 1, 1, 1)
    img1_coarse = img1_coarse.repeat(5, 1, 1, 1)
    img2_coarse = img2_coarse.repeat(5, 1, 1, 1)
    
    wimg2, _ = warp_hmg(img2, H_to_param(InverseBatch.apply(param_to_H(pt))))
    wimg2_coarse, _ = warp_hmg(img2_coarse, H_to_param(InverseBatch.apply(param_to_H(pt))))
    
    img1_n = normalize_img_batch(img1)
    wimg2_n = normalize_img_batch(wimg2)
    
    img1_coarse_n = normalize_img_batch(img1_coarse)
    wimg2_coarse_n = normalize_img_batch(wimg2_coarse)
    
    # Note: The following code would need a proper conv_net to work
    print("This demo code is incomplete - needs a proper conv_net for DeepLK")
    
if __name__ == "__main__":
    main()