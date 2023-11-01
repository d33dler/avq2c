from dataclasses import field

import torch

from models.attention.squeeze_excite.se_module import ClassRelatedAttentionModule
from models.backbones.base2d import BaseBackbone2d
from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.clustering.dn4_nbnn import I2C_KNN_AM
from models.utilities.utils import DataHolder, save_attention_map_as_image

torch.set_printoptions(profile="full")


##############################################################################
# Class: FourLayer_64F
##############################################################################

# Model: FourLayer_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21


class DN4_AM(BaselineBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data: DataHolder):
        super().__init__(data)
        self.attention = ClassRelatedAttentionModule(in_channels=64, reduction=data.cfg.BACKBONE.SE_REDUCTION,
                                                     round_activation=data.cfg.BACKBONE.get("SE_ROUND", True))
        self.knn = I2C_KNN_AM(self.knn.neighbor_k, attention_func= self.attention)

        del self.reg

    def forward(self):
        data = self.data
        data.q_F = self.features(data.q_in)
        data.S_F = self.features(data.S_in)
        attention_map = self.attention(data.q_F)

        qav_num, sav_num = (data.get_qv(), data.get_Sv()) if data.is_training() else (1, 1)
        data.sim_list = self.knn.forward(data.q_F, data.S_F, qav_num, sav_num,
                                         data.cfg.AUGMENTOR.STRATEGY if data.training else None,
                                         data.cfg.SHOT_NUM, attention_map=attention_map)
        # save_attention_map_as_image(data.q_CPU, attention_map.detach().cpu(), "tmp/attention_maps/")
        self.data.output = data.sim_list
        return data
