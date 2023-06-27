import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
import absl.flags as flags
from absl import app
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck

import skimage.io as io
import matplotlib.pyplot as plt

from torchvision import transforms
import torch
from Pascal3D import Pascal3D

import torch.nn.functional as F

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from network.resnet import resnet50, resnet101, ResnetHead
from network.resnet_backbone import ResNetBackboneNet
from network.resnet_rot_head import RotHeadNet
#from network.PoseR import Rot_red, Rot_green

#dataset_dir = '/data0/sunshichu/projects/Modified_6D_matrix_Fisher_distribution_for_probability_pose_estimation/datasets'
dataset_dir = '/data2/llq/distri/matrix_fisher/datasets/'


# Specification
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}

def get_pascal_no_warp_loaders(batch_size, train_all, voc_train):
    dataset = Pascal3D.Pascal3D(dataset_dir, train_all=train_all, use_warp=False, voc_train=voc_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset.get_train(False),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset.get_eval(),
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=False)
    return dataloader_train, dataloader_eval

class Fisher_n6d(nn.Module):
    def __init__(self, backbone, rot_head_net):
        super(Fisher_n6d, self).__init__()
        self.base = resnet101()
        self.backbone = backbone
        self.rot_head = rot_head_net

    def forward(self, im):
        feat = self.backbone(im) # feat = [bs, 2048, 8, 8], actually [bs, 512, 7, 7]
        #print("feat = ", feat.shape)
        # 6d normal vectors
        green_R_vec, red_R_vec = self.rot_head(feat)  # green_R_vec = [bs, 4, 56, 56]
        #print("green_R_vec = ", green_R_vec.shape)
        # normalization: green_R_normalized = [bs, 3, 56, 56]
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6) 
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        #print("green_R_normalized = ", p_green_R.shape)
        # sigmoid for confidence
        f_green_R = torch.sigmoid(green_R_vec[:, 0]) # f_green_R = [bs, 56, 56]
        f_red_R = torch.sigmoid(red_R_vec[:, 0])
        #print("green_R_confi = ", f_green_R.shape)
        return p_green_R, p_red_R, f_green_R, f_red_R
    
def main(argv):            
    arch = 'resnet'
    back_freeze = False
    back_input_channel = 3 # # channels of backbone's input
    back_layers_num = 34   # 18 | 34 | 50 | 101 | 152
    back_filters_num = 256  # number of filters for each layer
    
    rot_layers_num = 3
    rot_filters_num = 256 
    rot_conv_kernel_size = 3 
    rot_output_conv_kernel_size = 1
    rot_output_channels = 8 
    rot_head_freeze = True

    batch_size = 4
    train_all = True
    voc_train = False
    dataloader_train, dataloader_eval = get_pascal_no_warp_loaders(batch_size, train_all, voc_train)
    
    for image, extrinsic, class_idx_cpu, hard, _, _ in dataloader_train:
        im = image  # im.shape [bs, 3, 224, 224]
        class_idx=class_idx_cpu
        break
    # print("im = ", im.shape) # actually [bs, 3, 224, 224]
    
    base = resnet101()
    block_type, layers, in_channels, name = resnet_spec[back_layers_num]
    backbone = ResNetBackboneNet(block_type, layers, back_input_channel=3, back_freeze=False)
    feature = backbone.forward(im)  # features = [bs, 512, 7, 7]
    # print("feature = ", feature.shape) 
    
    rot_head_net = RotHeadNet(in_channels[-1], rot_layers_num=3, rot_filters_num=256, 
                              rot_conv_kernel_size=3, rot_output_conv_kernel_size=1,
                              rot_output_channels=8, rot_head_freeze=True)
    rot_green, rot_red = rot_head_net.forward(feature) # rot_green = [bs, 4, 56, 56]
    # print("rotation head green = ", rot_green.shape)

    model = Fisher_n6d(backbone, rot_head_net)
    p_green_R, p_red_R, f_green_R, f_red_R = model.forward(im)
    print("green vector = ", p_green_R.shape) # p_green_R = [bs, 3, 56, 56]
    print("green confidence = ", f_green_R.shape) # f_green_R = [bs, 56, 56]

if __name__ == "__main__":
    app.run(main)