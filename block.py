# -*- coding: utf-8 -*-
# @Author : xyoung
# @Time : 17:22  2023-05-14
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor
from timm.models.layers import DropPath


class PatchMerging(nn.Module):
    def __init__(self, fileter, ksize, stride, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2d(fileter, 2 * fileter, kernel_size=ksize, stride=stride, bias=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * fileter)
        else:
            self.norm = nn.Identity()
    
    def forward(self, x):
        x = self.norm(self.reduction(x))
        return x


class PConv(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError
    
    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x
    
    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Merging(nn.Module):
    def __init__(self, indim, outdim, ksize, stride):
        super().__init__()
        self.conv = nn.Conv2d(indim, outdim, kernel_size=ksize, stride=stride)
        self.bn = nn.BatchNorm2d(outdim)
    
    def forward(self, x):
        return self.bn(self.conv(x))


class FasterBlock(nn.Module):
    def __init__(self, dim, n_div, drop_path, Acti, forward="split_cat"):
        super().__init__()
        self.pconv1 = PConv(dim, n_div, forward)
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(dim, 2*dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2*dim),
            Acti()
        )
        self.pwconv3 = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1, bias=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x):
        x1 = self.pconv1(x)
        x1 = self.pwconv2(x1)
        x1 = self.pwconv3(x1)
        return x + self.drop_path(x1)


class BasicStage(nn.Module):
    def __init__(self, dim, depth, n_div, drop_path,
                 Acti):
        super().__init__()
        
        blocks_list = [
            FasterBlock(dim=dim, n_div=n_div, drop_path=drop_path, Acti=Acti)
            for i in range(depth)
        ]
        
        self.blocks = nn.Sequential(*blocks_list)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x
