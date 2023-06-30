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
from UPNA import UPNA

import torch.nn.functional as F

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from network.resnet import resnet50, resnet101, ResnetHead
from network.resnet_backbone import ResNetBackboneNet
from network.rot_head import RotHeadNet

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
        drop_last=True)
    return dataloader_train, dataloader_eval

class Fisher_n6d(nn.Module):
    def __init__(self, backbone, rot_head_net, n_classes, embedding_dim, num_hidden_nodes):
        super(Fisher_n6d, self).__init__()
        #self.backbone = backbone
        
        self.out_dim = 9
        self.base = resnet101(pretrained=True, progress=True)
        self.backbone = ResnetHead(self.base, n_classes, embedding_dim, num_hidden_nodes, self.out_dim)
        
        self.rot_head = rot_head_net

    def forward(self, im, class_idx):
        feat = self.backbone(im, class_idx) # feat = net_F = [bs, 9]
       
        # 6d normal vectors
        net_F, green_R_vec, red_R_vec = self.rot_head(feat)  # green_R_vec = [bs, 4]
        # normalization: green_R_normalized = [bs, 3]
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6) 
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = torch.sigmoid(green_R_vec[:, 0]) # f_green_R = [bs]
        f_red_R = torch.sigmoid(red_R_vec[:, 0])
        return feat, net_F, p_green_R, p_red_R, f_green_R, f_red_R
    
def main(argv):     
    num_classes=13
    embedding_dim=32
    num_hidden_nodes=512
           
    back_freeze = False
    back_input_channel = 3 # # channels of backbone's input
    back_layers_num = 101   # 18 | 34 | 50 | 101 | 152
    
    rot_layers_num = 3
    rot_filters_num = 256 
    rot_conv_kernel_size = 3 
    rot_output_conv_kernel_size = 1
    rot_output_channels = 17 
    rot_head_freeze = True

    batch_size = 32
    train_all = True
    voc_train = False
    dataloader_train, dataloader_eval = get_pascal_no_warp_loaders(batch_size, train_all, voc_train)
    
    for image, extrinsic, class_idx_cpu, hard, _, _ in dataloader_train:
        im = image  # im.shape [bs, 3, 224, 224]
        class_idx=class_idx_cpu
        break
    # print("im = ", im.shape) # actually [bs, 3, 224, 224]
    
    block_type, layers, in_channels, name = resnet_spec[back_layers_num]
    backbone = ResNetBackboneNet(block_type, layers, back_input_channel, back_freeze)
    feature = backbone.forward(im)  # features = [bs, 512, 7, 7]
    # print("feature = ", feature.shape) 
    
    rot_head_net = RotHeadNet(in_channels[-1], rot_layers_num, rot_filters_num, 
                              rot_conv_kernel_size, rot_output_conv_kernel_size,
                              rot_output_channels, rot_head_freeze)
    net_F, rot_green, rot_red = rot_head_net.forward(feature) # rot_green = [bs, 4]
    # print("rotation head green = ", rot_green.shape)

    model = Fisher_n6d(backbone, rot_head_net, num_classes, embedding_dim, num_hidden_nodes)
    feat, net_F, p_green_R, p_red_R, f_green_R, f_red_R = model.forward(im, class_idx)
    # print("green vector = ", p_green_R.shape) # p_green_R = [bs, 3]
    # print("green confidence = ", f_green_R.shape) # f_green_R = [bs]
    # print("net_F = ", net_F.shape) # net_F = [bs, 9]

if __name__ == "__main__":
    app.run(main)