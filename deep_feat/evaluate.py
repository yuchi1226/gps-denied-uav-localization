import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import os
import random
import DeepLKBatch as dlk
import glob
from math import cos, sin, pi, sqrt
import time
import sys
import gc
import numpy as np
import argparse

# USAGE:
# python3 evaluate.py MODE FOLDER_NAME DATAPATH MODEL_PATH VGG_MODEL_PATH --TEST_DATA_SAVE_PATH

# TRAIN:
# python3 evaluate.py train woodbridge ../sat_data/ trained_model_output.pth ../models/vgg16_model.pth

# TEST:
# python3 evaluate.py test woodbridge ../sat_data/ ../models/conv_02_17_18_1833.pth ../models/vgg16_model.pth -t test_out.txt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("MODE")
    parser.add_argument("FOLDER_NAME")
    parser.add_argument("DATAPATH")
    parser.add_argument("MODEL_PATH")
    parser.add_argument("VGG_MODEL_PATH")
    parser.add_argument("-t", "--TEST_DATA_SAVE_PATH")
    
    args = parser.parse_args()
    
    MODE = args.MODE
    FOLDER_NAME = args.FOLDER_NAME
    FOLDER = FOLDER_NAME + '/'
    DATAPATH = args.DATAPATH
    MODEL_PATH = args.MODEL_PATH
    VGG_MODEL_PATH = args.VGG_MODEL_PATH
    
    if MODE == 'test':
        if args.TEST_DATA_SAVE_PATH is None:
            exit('Must supply TEST_DATA_SAVE_PATH argument in test mode')
        else:
            TEST_DATA_SAVE_PATH = args.TEST_DATA_SAVE_PATH

# 設備設置
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# 訓練/測試參數
min_scale = 0.75
max_scale = 1.25
angle_range = 15  # degrees
projective_range = 0
translation_range = 10  # pixels
lower_sz = 200  # pixels, square
upper_sz = 220
warp_pad = 0.4
training_sz = 175
training_sz_pad = round(training_sz + training_sz * 2 * warp_pad)

def random_param_generator():
    scale = random.uniform(min_scale, max_scale)
    angle = random.uniform(-angle_range, angle_range)
    projective_x = random.uniform(-projective_range, projective_range)
    projective_y = random.uniform(-projective_range, projective_range)
    translation_x = random.uniform(-translation_range, translation_range)
    translation_y = random.uniform(-translation_range, translation_range)
    
    rad_ang = angle / 180 * pi
    
    p_gt = torch.Tensor([scale + cos(rad_ang) - 2,
                       -sin(rad_ang),
                       translation_x,
                       sin(rad_ang),
                       scale + cos(rad_ang) - 2,
                       translation_y,
                       projective_x, 
                       projective_y]).to(device)
    
    p_gt = p_gt.view(8, 1)
    p_gt = p_gt.unsqueeze(0)  # Add batch dimension
    return p_gt

def static_data_generator(batch_size):
    FOLDERPATH = os.path.join(DATAPATH, FOLDER, 'images/')
    images_dir = glob.glob(FOLDERPATH + '*.png')
    random.shuffle(images_dir)
    
    if len(images_dir) < 2:
        raise ValueError("Need at least 2 images in the directory")
        
    img = Image.open(images_dir[0])
    template = Image.open(images_dir[1])
    
    in_W, in_H = img.size
    
    img_batch = torch.zeros(batch_size, 3, training_sz, training_sz, device=device)
    template_batch = torch.zeros(batch_size, 3, training_sz, training_sz, device=device)
    param_batch = torch.zeros(batch_size, 8, 1, device=device)
    
    for i in range(batch_size):
        seg_sz = 200
        seg_sz_pad = round(seg_sz + seg_sz * 2 * warp_pad)
        
        loc_x = 40
        loc_y = 40
        
        img_seg_pad = img.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        img_seg_pad = img_seg_pad.resize((training_sz_pad, training_sz_pad))
        
        template_seg_pad = template.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        template_seg_pad = template_seg_pad.resize((training_sz_pad, training_sz_pad))
        
        img_seg_pad = transforms.ToTensor()(img_seg_pad).to(device)
        template_seg_pad = transforms.ToTensor()(template_seg_pad).to(device)
        
        scale = 1.2
        angle = 10
        projective_x = 0
        projective_y = 0
        translation_x = -5
        translation_y = 10
        
        rad_ang = angle / 180 * pi
        
        p_gt = torch.Tensor([scale + cos(rad_ang) - 2,
                           -sin(rad_ang),
                           translation_x,
                           sin(rad_ang),
                           scale + cos(rad_ang) - 2,
                           translation_y,
                           projective_x, 
                           projective_y]).to(device)
        
        p_gt = p_gt.view(8, 1)
        p_gt = p_gt.unsqueeze(0)  # Add batch dimension
        
        img_seg_pad_w, _ = dlk.warp_hmg(
            img_seg_pad.unsqueeze(0),
            dlk.H_to_param(dlk.InverseBatch.apply(dlk.param_to_H(p_gt)))
        )
        
        img_seg_pad_w = img_seg_pad_w.squeeze(0)
        
        pad_side = round(training_sz * warp_pad)
        
        img_seg_w = img_seg_pad_w[:,
                        pad_side : pad_side + training_sz,
                        pad_side : pad_side + training_sz]
        
        template_seg = template_seg_pad[:,
                        pad_side : pad_side + training_sz,
                        pad_side : pad_side + training_sz]
        
        img_batch[i, :, :, :] = img_seg_w
        template_batch[i, :, :, :] = template_seg
        param_batch[i, :, :] = p_gt[0, :, :]
        
    return img_batch, template_batch, param_batch

def data_generator(batch_size):
    FOLDERPATH = os.path.join(DATAPATH, FOLDER, 'images/')
    images_dir = glob.glob(FOLDERPATH + '*.png')
    
    if len(images_dir) < 2:
        raise ValueError("Need at least 2 images in the directory")
        
    random.shuffle(images_dir)
    img = Image.open(images_dir[0])
    template = Image.open(images_dir[1])
    
    in_W, in_H = img.size
    
    img_batch = torch.zeros(batch_size, 3, training_sz, training_sz, device=device)
    template_batch = torch.zeros(batch_size, 3, training_sz, training_sz, device=device)
    param_batch = torch.zeros(batch_size, 8, 1, device=device)
    
    for i in range(batch_size):
        seg_sz = random.randint(lower_sz, upper_sz)
        seg_sz_pad = round(seg_sz + seg_sz * 2 * warp_pad)
        
        loc_x = random.randint(0, (in_W - seg_sz_pad) - 1)
        loc_y = random.randint(0, (in_H - seg_sz_pad) - 1)
        
        img_seg_pad = img.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        img_seg_pad = img_seg_pad.resize((training_sz_pad, training_sz_pad))
        
        template_seg_pad = template.crop((loc_x, loc_y, loc_x + seg_sz_pad, loc_y + seg_sz_pad))
        template_seg_pad = template_seg_pad.resize((training_sz_pad, training_sz_pad))
        
        img_seg_pad = transforms.ToTensor()(img_seg_pad).to(device)
        template_seg_pad = transforms.ToTensor()(template_seg_pad).to(device)
        
        scale = random.uniform(min_scale, max_scale)
        angle = random.uniform(-angle_range, angle_range)
        projective_x = random.uniform(-projective_range, projective_range)
        projective_y = random.uniform(-projective_range, projective_range)
        translation_x = random.uniform(-translation_range, translation_range)
        translation_y = random.uniform(-translation_range, translation_range)
        
        rad_ang = angle / 180 * pi
        
        p_gt = torch.Tensor([scale + cos(rad_ang) - 2,
                           -sin(rad_ang),
                           translation_x,
                           sin(rad_ang),
                           scale + cos(rad_ang) - 2,
                           translation_y,
                           projective_x, 
                           projective_y]).to(device)
        
        p_gt = p_gt.view(8, 1)
        p_gt = p_gt.unsqueeze(0)  # Add batch dimension
        
        img_seg_pad_w, _ = dlk.warp_hmg(
            img_seg_pad.unsqueeze(0),
            dlk.H_to_param(dlk.InverseBatch.apply(dlk.param_to_H(p_gt)))
        )
        
        img_seg_pad_w = img_seg_pad_w.squeeze(0)
        
        pad_side = round(training_sz * warp_pad)
        
        img_seg_w = img_seg_pad_w[:,
                        pad_side : pad_side + training_sz,
                        pad_side : pad_side + training_sz]
        
        template_seg = template_seg_pad[:,
                        pad_side : pad_side + training_sz,
                        pad_side : pad_side + training_sz]
        
        img_batch[i, :, :, :] = img_seg_w
        template_batch[i, :, :, :] = template_seg
        param_batch[i, :, :] = p_gt[0, :, :]
        
    return img_batch, template_batch, param_batch

def corner_loss(p, p_gt):
    batch_size, _, _ = p.size()
    
    H_p = dlk.param_to_H(p)
    H_gt = dlk.param_to_H(p_gt)
    
    corners = torch.Tensor([[-training_sz_pad/2, training_sz_pad/2, training_sz_pad/2, -training_sz_pad/2],
                           [-training_sz_pad/2, -training_sz_pad/2, training_sz_pad/2, training_sz_pad/2],
                           [1, 1, 1, 1]]).to(device)
    
    corners = corners.repeat(batch_size, 1, 1)
    
    corners_w_p = H_p.bmm(corners)
    corners_w_gt = H_gt.bmm(corners)
    
    corners_w_p = corners_w_p[:, 0:2, :] / corners_w_p[:, 2:3, :]
    corners_w_gt = corners_w_gt[:, 0:2, :] / corners_w_gt[:, 2:3, :]
    
    loss = ((corners_w_p - corners_w_gt) ** 2).sum()
    return loss

def test():
    dlk_vgg16 = dlk.DeepLK(dlk.vgg16Conv(VGG_MODEL_PATH)).to(device)
    dlk_trained = dlk.DeepLK(dlk.custom_net(MODEL_PATH)).to(device)
    
    testbatch_sz = 1  # keep as 1 for accurate corner error computation
    test_rounds_num = 50
    rounds_per_pair = 50
    
    test_results = np.zeros((test_rounds_num, 5), dtype=float)
    
    print('Testing...')
    print('TEST DATA SAVE PATH: ', TEST_DATA_SAVE_PATH)
    print('DATAPATH: ', DATAPATH)
    print('FOLDER: ', FOLDER)
    print('MODEL PATH: ', MODEL_PATH)
    print('USE CUDA: ', USE_CUDA)
    print('min_scale: ', min_scale)
    print('max_scale: ', max_scale)
    print('angle_range: ', angle_range)
    print('projective_range: ', projective_range)
    print('translation_range: ', translation_range)
    print('lower_sz: ', lower_sz)
    print('upper_sz: ', upper_sz)
    print('warp_pad: ', warp_pad)
    print('test batch size: ', testbatch_sz, ' number of test round: ', test_rounds_num, ' rounds per pair: ', rounds_per_pair)
    
    img_test_data = torch.zeros(test_rounds_num, 3, training_sz, training_sz, device=device)
    template_test_data = torch.zeros(test_rounds_num, 3, training_sz, training_sz, device=device)
    param_test_data = torch.zeros(test_rounds_num, 8, 1, device=device)
    
    for i in range(test_rounds_num // rounds_per_pair):
        print('gathering data...', i+1, ' / ', test_rounds_num // rounds_per_pair)
        batch_index = i * rounds_per_pair
        
        img_batch, template_batch, param_batch = data_generator(rounds_per_pair)
        
        img_test_data[batch_index:batch_index + rounds_per_pair] = img_batch
        template_test_data[batch_index:batch_index + rounds_per_pair] = template_batch
        param_test_data[batch_index:batch_index + rounds_per_pair] = param_batch
        
        sys.stdout.flush()
    
    for i in range(test_rounds_num):
        img_batch_unnorm = img_test_data[i].unsqueeze(0)
        template_batch_unnorm = template_test_data[i].unsqueeze(0)
        param_batch = param_test_data[i].unsqueeze(0)
        
        img_batch = dlk.normalize_img_batch(img_batch_unnorm)
        template_batch = dlk.normalize_img_batch(template_batch_unnorm)
        
        img_batch_coarse = nn.AvgPool2d(4)(img_batch)
        template_batch_coarse = nn.AvgPool2d(4)(template_batch)
        
        vgg_param, _ = dlk_vgg16(img_batch, template_batch, tol=1e-3, max_itr=70, conv_flag=1)
        trained_param, _ = dlk_trained(img_batch, template_batch, tol=1e-3, max_itr=70, conv_flag=1)
        coarse_param, _ = dlk_vgg16(img_batch_coarse, template_batch_coarse, tol=1e-3, max_itr=70, conv_flag=0)
        
        vgg_loss = corner_loss(vgg_param, param_batch)
        no_op_loss = corner_loss(torch.zeros(testbatch_sz, 8, 1, device=device), param_batch)
        trained_loss = corner_loss(trained_param, param_batch)
        coarse_loss = corner_loss(coarse_param, param_batch)
        
        test_results[i, 0] = vgg_loss.item()
        test_results[i, 1] = trained_loss.item()
        test_results[i, 2] = coarse_loss.item()
        test_results[i, 3] = no_op_loss.item()
        
        print('test: ', i, 
              ' vgg16 loss: ', round(sqrt(float(vgg_loss.item())/4), 2), 
              ' trained loss: ', round(sqrt(float(trained_loss.item())/4), 2), 
              ' coarse pix loss: ', round(sqrt(float(coarse_loss.item())/4), 2), 
              ' no-op loss: ', round(sqrt(float(no_op_loss.item())/4), 2))
        
        sys.stdout.flush()
    
    np.savetxt(TEST_DATA_SAVE_PATH, test_results, delimiter=',')

def train():
    dlk_net = dlk.DeepLK(dlk.vgg16Conv(VGG_MODEL_PATH)).to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, dlk_net.conv_func.parameters()), lr=0.0001)
    
    best_valid_loss = float('inf')
    
    minibatch_sz = 10
    #num_minibatch = 25000
    num_minibatch = 50
    valid_batch_sz = 10
    valid_num_generator = 50
    
    print('Training...')
    print('DATAPATH: ', DATAPATH)
    print('FOLDER: ', FOLDER)
    print('MODEL_PATH: ', MODEL_PATH)
    print('VGG MODEL PATH', VGG_MODEL_PATH)
    print('USE CUDA: ', USE_CUDA)
    print('min_scale: ', min_scale)
    print('max_scale: ', max_scale)
    print('angle_range: ', angle_range)
    print('projective_range: ', projective_range)
    print('translation_range: ', translation_range)
    print('lower_sz: ', lower_sz)
    print('upper_sz: ', upper_sz)
    print('warp_pad: ', warp_pad)
    print('training_sz: ', training_sz_pad)
    print('minibatch size: ', minibatch_sz, ' number of minibatches: ', num_minibatch)
    
    img_train_data = torch.zeros(num_minibatch, 3, training_sz, training_sz, device=device)
    template_train_data = torch.zeros(num_minibatch, 3, training_sz, training_sz, device=device)
    param_train_data = torch.zeros(num_minibatch, 8, 1, device=device)
    
    for i in range(num_minibatch // minibatch_sz):
        print('gathering training data...', i+1, ' / ', num_minibatch // minibatch_sz)
        batch_index = i * minibatch_sz
        
        img_batch, template_batch, param_batch = data_generator(minibatch_sz)
        
        img_train_data[batch_index:batch_index + minibatch_sz] = img_batch
        template_train_data[batch_index:batch_index + minibatch_sz] = template_batch
        param_train_data[batch_index:batch_index + minibatch_sz] = param_batch
        sys.stdout.flush()
    
    valid_img_batch = torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz, device=device)
    valid_template_batch = torch.zeros(valid_batch_sz * valid_num_generator, 3, training_sz, training_sz, device=device)
    valid_param_batch = torch.zeros(valid_batch_sz * valid_num_generator, 8, 1, device=device)
    
    for i in range(valid_num_generator):
        print('gathering validation data...', i+1, ' / ', valid_num_generator)
        img_batch, template_batch, param_batch = data_generator(valid_batch_sz)
        valid_img_batch[i*valid_batch_sz:(i+1)*valid_batch_sz] = img_batch
        valid_template_batch[i*valid_batch_sz:(i+1)*valid_batch_sz] = template_batch
        valid_param_batch[i*valid_batch_sz:(i+1)*valid_batch_sz] = param_batch
    
    for i in range(num_minibatch):
        start = time.time()
        optimizer.zero_grad()
        
        img_batch_unnorm = img_train_data[i].unsqueeze(0)
        template_batch_unnorm = template_train_data[i].unsqueeze(0)
        training_param_batch = param_train_data[i].unsqueeze(0)
        
        training_img_batch = dlk.normalize_img_batch(img_batch_unnorm)
        training_template_batch = dlk.normalize_img_batch(template_batch_unnorm)
        
        dlk_param_batch, _ = dlk_net(training_img_batch, training_template_batch, tol=1e-3, max_itr=1, conv_flag=1)
        
        loss = corner_loss(dlk_param_batch, training_param_batch)
        loss.backward()
        optimizer.step()
        
        dlk_valid_param_batch, _ = dlk_net(valid_img_batch, valid_template_batch, tol=1e-3, max_itr=1, conv_flag=1)
        valid_loss = corner_loss(dlk_valid_param_batch, valid_param_batch)
        
        print('mb: ', i+1, ' training loss: ', float(loss.item()), ' validation loss: ', float(valid_loss.item()), end='')
        
        if (i == 0) or (float(valid_loss.item()) < float(best_valid_loss)):
            best_valid_loss = valid_loss.item()
            torch.save(dlk_net.conv_func, MODEL_PATH)
            print(' best validation loss: ', float(best_valid_loss), ' (saving)')
        else:
            print(' best validation loss: ', float(best_valid_loss))
        
        gc.collect()
        sys.stdout.flush()
        
        end = time.time()

if __name__ == "__main__":
    print('PID: ', os.getpid())
    
    if MODE == 'test': #type:ignore
        test()
    else:
        train()