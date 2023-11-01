from dataclasses import field

import torch

from models.attention.mh_sa.multihead_attention import MultiHeadAttentionModule
from models.backbones.base2d import BaseBackbone2d
from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.utilities.utils import DataHolder

torch.set_printoptions(profile="full")


##############################################################################
# Class: FourLayer_64F
##############################################################################

# Model: DN4 Multihead Attention
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class DN4_MATT(BaselineBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data: DataHolder):
        super().__init__(data)
        self.attention = MultiHeadAttentionModule(64, num_heads=4)

        del self.reg

    def forward(self):
        data = self.data
        data.q_F = self.features(data.q_in)
        data.S_F = self.features(data.S_in)
        self.data = self.attention(data.q_F)

        qav_num, sav_num = (data.get_qv(), data.get_Sv()) if data.is_training() else (1, 1)
        data.sim_list = self.knn.forward(data.q_F, data.S_F, qav_num, sav_num,
                                         data.cfg.AUGMENTOR.STRATEGY if data.training else None,
                                         data.cfg.SHOT_NUM)
        # save_attention_map_as_image(data.q_CPU, attention_map.detach().cpu(), "tmp/attention_maps/")
        self.data.output = data.sim_list
        return data
