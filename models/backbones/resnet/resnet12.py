import math
from dataclasses import field

import torch.nn as nn

from models.backbones.backbones_utils import ResNet12
from models.backbones.base2d import BaseBackbone2d
from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d


##############################################################################
# Classes: ResNetLike
##############################################################################

# Model: ResNetBackbone2d
# Refer to: https://github.com/gidariss/FewShotWithoutForgetting
# Input: One query image and a support set
# Base_model: 4 ResBlock layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84, for miniImageNet
# Filters: 64->x->x->x


class ResNetBackbone2d(BaselineBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data):  # neighbor_k=3
        super().__init__(data)

        self.features = ResNet12()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self):
        return super().forward()
