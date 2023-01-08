import random

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
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from point_4d_convolution import *
from transformer import *
# from pst_convolutions import *
# from pointnet import pointnet




class PrimitiveTransformer(nn.Module):
    def __init__(self, radius, nsamples, spatial_stride,                                # P4DConv: spatial
                 temporal_kernel_size, temporal_stride,                                 # P4DConv: temporal
                 emb_relu,                                                              # embedding: relu
                 dim, depth, heads, dim_head,                                           # transformer
                 mlp_dim, num_classes):                                                 # output
        super().__init__()

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.emb_relu1 = nn.ReLU()
        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim)
        self.emb_relu2 = nn.ReLU()
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim)

    def forward(self, input):


        # 4d BACKBONE
        # [B, L, N, 3]
        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]

        features = features.transpose(2, 3)  # B ,L , n, C
        B, L, N, C = features.size()

        raw_feat = features

        point_feat = torch.reshape(input=raw_feat, shape=(B * L * 8, -1, C))  # [B*L*4, n', C]
        point_feat = self.emb_relu1(point_feat)
        point_feat = self.transformer1(point_feat)  # [B*L*4, n', C]

        primitive_feature = point_feat.permute(0, 2, 1)
        primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))  # B*l*4, C, 1
        primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L * 8, C))  # [B, L*4, C]

        anchor_feature = torch.reshape(input=primitive_feature, shape=(B*L, 8, C))
        anchor_feature = anchor_feature.permute(0, 2, 1)
        anchor_feature = F.adaptive_max_pool1d(anchor_feature, (1))
        anchor_feature = torch.reshape(input=anchor_feature, shape=(B, L, C))  

        primitive_feature = self.emb_relu2(anchor_feature)
        primitive_feature = self.transformer2(primitive_feature) # B. L*4, C

        # primitive_feature = torch.reshape(input=primitive_feature, shape=(B*L, 8, C))
        # primitive_feature = primitive_feature.permute(0, 2, 1)
        # primitive_feature = F.adaptive_max_pool1d(primitive_feature, (1))
        # primitive_feature = torch.reshape(input=primitive_feature, shape=(B, L, C))  

        return anchor_feature, primitive_feature

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