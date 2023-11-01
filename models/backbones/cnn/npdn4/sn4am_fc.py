import torch.nn as nn
import torchvision.ops
from easydict import EasyDict

from models.backbones.cnn.npdn4.sn4_fc import B_4L64F_MCNP
from models.utilities.utils import DataHolder, weights_init_normal, \
    weights_init_kaiming

"""
NOTE: Not used in the paper
"""
##############################################################################
# Class: SN4_AM_FC
##############################################################################

# Model: Siamese_FourLayer_64F
# Input: Query set, Support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class B_4L64F_AM_MCNP(B_4L64F_MCNP):
    """
    Attention Module (AM) + Multi-class N-Pair Loss (MCNP)
    """

    def __init__(self, data: DataHolder, config: EasyDict = None):
        super().__init__(data)
        norm_layer = self.norm_layer
        self.features = nn.Sequential(  # 3*84*84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*42*42

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21

            norm_layer(64),
            nn.LeakyReLU(0.2, True),  # 64*21*21
        )
        self.fc = nn.Sequential(nn.Flatten(start_dim=1),
                                nn.Linear(64 * 21 * 21, self.output_shape)
                                )
        self.features.apply(weights_init_kaiming)
        self.fc.apply(weights_init_kaiming)

    def forward(self):
        return super().forward()
