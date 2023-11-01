import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F, init
import matplotlib.pyplot as plt
import os

from models.utilities.utils import net_init_weights_normal


class ClassRelatedAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16, round_activation=True):
        super(ClassRelatedAttentionModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.conv_Wg = nn.Conv2d(in_channels, in_channels // reduction,
                                 kernel_size=1, bias=False)  # no reduction in paper (CC x H x W)
        self.conv_Wk = nn.Conv2d(in_channels, in_channels // reduction,
                                 kernel_size=1, bias=False)  # reduction in paper to ( 1 x H x W)
        self.fc1 = nn.Conv2d(in_channels // reduction, in_channels // reduction,
                             kernel_size=1, bias=False)  # input in paper = (CC x H x W)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.round_activation = round_activation
        net_init_weights_normal(self)

    def nullify_borders(self, tensor, border_size=4):
        """
        Set the values of the borders of a tensor to zero.

        :param tensor: Input tensor of shape (..., H, W).
        :param border_size: Size of the border to nullify.
        :return: Tensor with nullified borders.
        """
        tensor[..., :border_size, :] = 0  # Top border
        tensor[..., -border_size:, :] = 0  # Bottom border
        tensor[..., :, :border_size] = 0  # Left border
        tensor[..., :, -border_size:] = 0  # Right border
        return tensor

    def forward(self, x):
        b, c, h, w = x.size()
        # Non-local operation
        fg = self.conv_Wg(x)
        fk = self.conv_Wk(x)
        fk = F.softmax(fk.view(b, self.in_channels // self.reduction, -1), dim=-1).view(b,
                                                                                       self.in_channels // self.reduction,
                                                                                       h, w)
        non_local_op = fg * fk
        non_local_op = non_local_op.sum(dim=[2, 3])
        # Excitation operation
        y = self.fc1(non_local_op.unsqueeze(-1).unsqueeze(-1))
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)

        # Apply the weight vector to the input feature map
        y = torch.round(y) if self.round_activation else y
        weighted_x = x * y

        # Sum the features of the scene-related channels
        scene_related_features = weighted_x.sum(dim=1, keepdim=True)
        # Obtain the scene-class-related attention feature map
        attention_feature_map = self.sigmoid(scene_related_features)
        # self.save_attention_map_as_image(attention_feature_map.detach().cpu().numpy(), 'tmp/attention_maps/')
        #print(attention_feature_map.size())
        return self.nullify_borders(attention_feature_map)
