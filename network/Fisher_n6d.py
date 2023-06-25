import torch
import torch.nn as nn
import numpy as np
import absl.flags as flags
from absl import app

import torch.nn.functional as F

import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-2])
sys.path.append(root_path)

from network.resnet import resnet50, resnet101, ResnetHead
from network.PoseR import Rot_red, Rot_green

class Fisher_n6d(nn.Module):
    def __init__(self, base, n_classes, embedding_dim, num_hidden_nodes, n_out):
        super(Fisher_n6d, self).__init__()
        self.base = resnet101()
        self.resnethead = ResnetHead(self.base, n_classes, embedding_dim, 512, n_out)
        self.rot_green = Rot_green()
        self.rot_red = Rot_red()

    def forward(self, n_classes, embedding_dim, n_out):
        feat = self.resnethead(self.base, n_classes, embedding_dim, 512, n_out)
        print("ResnetHead output = ", feat.shape())

        # 6d normal vectors
        green_R_vec =  self.rot_green(feat.permute(0, 2, 1))  # b x 4
        red_R_vec =  self.rot_red(feat.permute(0, 2, 1))  # b x 4
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])
        
        return p_green_R, p_red_R, f_green_R, f_red_R
    
def main(argv):
    points = torch.rand(2, 1286, 1500)  # batchsize x feature x numofpoint
    
    im = '/data2/llq/distri/matrix_fisher/datasets/Pascal3d/Images/boat_pascal/2008_000120.jpg'
    class_idx =  13
    
    base = resnet50(im)
    n_classes = 13 # for pascal
    embedding_dim = 32 # for pascal3d_no_augment.json
    n_out = 9
    
    # rot_head_red = Rot_red()
    # print("rotation head red = ", rot_head_red.shape())
    
    feature = ResnetHead(base, n_classes, embedding_dim, 512, n_out)
    output = feature.forward(im, class_idx)
    print("ResnetHead output = ", output)

    # p_green_R, p_red_R, f_green_R, f_red_R = Fisher_n6d(base, n_classes, embedding_dim, 512, n_out)
    # print("green vector = ", p_green_R.shape())


if __name__ == "__main__":
    app.run(main)
