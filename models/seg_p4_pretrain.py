import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

import modules.pointnet2_utils
from modules.point_4d_convolution import *
from modules.transformer import *

class P4Transformer(nn.Module):
    def __init__(self, radius=0.9, nsamples=32, num_classes=49):
        super(P4Transformer, self).__init__()

        self.conv1 = P4DConv(in_planes=3,
                             mlp_planes=[32,64,128],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.conv2 = P4DConv(in_planes=128,
                             mlp_planes=[128, 128, 256],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.conv3 = P4DConv(in_planes=256,
                             mlp_planes=[256,256,512],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*radius, nsamples],
                             temporal_kernel_size=3,
                             spatial_stride=4,
                             temporal_stride=1,
                             temporal_padding=[1,1])

        self.conv4 = P4DConv(in_planes=512,
                             mlp_planes=[512,512,1024],
                             mlp_batch_norm=[True, True, True],
                             mlp_activation=[True, True, True],
                             spatial_kernel_size=[2*2*2*radius, nsamples],
                             temporal_kernel_size=1,
                             spatial_stride=2,
                             temporal_stride=1,
                             temporal_padding=[0,0])

        self.emb_relu = nn.ReLU()
        self.transformer = Transformer(dim=1024, depth=2, heads=4, dim_head=256, mlp_dim=1024)


    def forward(self, xyzs, rgbs):

        device = 'cuda'

        new_xyzs1, new_features1 = self.conv1(xyzs, rgbs)

        new_xyzs2, new_features2 = self.conv2(new_xyzs1, new_features1)

        new_xyzs3, new_features3 = self.conv3(new_xyzs2, new_features2)

        new_xyzs4, new_features4 = self.conv4(new_xyzs3, new_features3)

        B, L, C, N = new_features4.size()

        features = new_features4.permute(0, 1, 3, 2)                                                                                        # [B, L, n2, C]
        features = torch.reshape(input=features, shape=(features.shape[0], features.shape[1]*features.shape[2], features.shape[3]))         # [B, L*n2, C]

        raw_one = features.reshape(B, L, N, C)
        raw_one = raw_one.reshape(B*L, N, C)
        raw_one = raw_one.permute(0,2,1)
        raw_one = F.adaptive_max_pool1d(raw_one, (1)).reshape(B, L, C)

        features = self.emb_relu(features)
        features = self.transformer(features)

        trans_one = features.reshape(B, L, N, C)
        trans_one = trans_one.reshape(B*L, N, C)
        trans_one = trans_one.permute(0,2,1)
        trans_one = F.adaptive_max_pool1d(trans_one, (1)).reshape(B, L, C)
    
        return raw_one, trans_one


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


