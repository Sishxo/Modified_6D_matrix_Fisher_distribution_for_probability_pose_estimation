import torch
import torch.nn as nn


class Fisher_n6d(nn.Module):
    def __init__(self, resnet_head, fisher_head, rot_head, batch_size):
        super(Fisher_n6d, self).__init__()
        self.base_net = resnet_head
        self.fisher_head = fisher_head
        self.rot_head = rot_head
        self.batch_size = batch_size

    def forward(self, x, class_idx):
        features = self.base_net(x)
        fisher_output = self.fisher_head(features, class_idx)
        feat = features.view(self.batch_size, 32, 8, 8)  # feature=[bs, 32, 8, 8]
        feat = feat.repeat(1, 64, 1, 1)  # feature=[bs, 2048, 8, 8]
        green_R_vec, red_R_vec = self.rot_head(feat)

        return fisher_output, green_R_vec, red_R_vec
