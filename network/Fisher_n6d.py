import sys
sys.path.append("..")

import torch
import torch.nn as nn
import numpy as np
import absl.flags as flags
from absl import app
from PIL import Image
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
from network.PoseR import Rot_red, Rot_green

dataset_dir = '/data0/sunshichu/projects/Modified_6D_matrix_Fisher_distribution_for_probability_pose_estimation/datasets'

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
    
    base = resnet50()
    n_classes = 13 # for pascal
    embedding_dim = 32 # for pascal3d_no_augment.json
    n_out = 9
    
    batch_size = 4
    train_all = True
    voc_train = False
    dataloader_train, dataloader_eval = get_pascal_no_warp_loaders(batch_size, train_all, voc_train)
    
    for image, extrinsic, class_idx_cpu, hard, _, _ in dataloader_train:
        im = image
        print(im.shape)
        class_idx=class_idx_cpu
        break
    
    print(class_idx.shape)
    feature = ResnetHead(base, n_classes, embedding_dim, 512, n_out)
    output = feature.forward(im, class_idx)
    print("ResnetHead output = ", output.shape)
    print(output)
    #input()
    
    # rot_head_red = Rot_red()
    # print("rotation head red = ", rot_head_red.shape())

    # p_green_R, p_red_R, f_green_R, f_red_R = Fisher_n6d(base, n_classes, embedding_dim, 512, n_out)
    # print("green vector = ", p_green_R.shape())


if __name__ == "__main__":
    app.run(main)
