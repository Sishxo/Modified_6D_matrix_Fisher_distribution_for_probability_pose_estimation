import torch
import torch.nn as nn


class Fisher_n6d(nn.Module):
    def __init__(self, resnet_head, fisher_head):
        super(Fisher_n6d,self).__init__()
        self.base_net=resnet_head
        self.fisher_head = fisher_head
        #self.rot_head_net=rot_head

    def forward(self, x, class_idx):
        features = self.base_net(x)
        fisher_output = self.fisher_head(features,class_idx)
        #rotation_output = self.rot_head(features)
        
        return fisher_output#,rotation_output
        
        
    