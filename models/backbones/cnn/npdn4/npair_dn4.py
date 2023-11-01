from dataclasses import field

import torch
from easydict import EasyDict
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

from models.backbones.base2d import BaseBackbone2d
from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.clustering import I2C_KNN
from models.utilities.custom_loss import NPairMCLoss, NPairAngularLoss, NPairMCLossLSE
from models.utilities.utils import DataHolder


##############################################################################
# Class: SiameseNetworkKNN
##############################################################################

# Model: Siamese_FourLayer_64F_KNN
# Input: 3 x 84 x 84
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet, Stanford Dogs, Stanford Cars, CUB-200-2011)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class DN4_MCNP(BaselineBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data: DataHolder):
        super().__init__(data)
        self.criterion = NPairMCLoss().cuda()
        self.eval_criterion = CrossEntropyLoss().cuda()

    def forward(self):
        data = self.data
        data.q_F = self.features(data.q_in)
        data.S_F = self.features(data.S_in)
        qav_num, sav_num = (data.get_qv(), data.get_Sv()) if data.is_training() else (1, 1)
        data.sim_list = self.knn.forward(data.q_F, data.S_F, qav_num, sav_num,
                                                   data.cfg.AUGMENTOR.STRATEGY if data.training else None,
                                                   data.cfg.SHOT_NUM)
        self.data.output = data.sim_list
        if data.is_training():
            B, S = data.sim_list.shape
            # Normalize the batch tensor column-wise so that the scores are within [-1, 1]
            data.sim_list = softmax(data.sim_list, dim=1)
            # Assuming you have a targets 1-d vector that specifies the index of the positive class
            targets = data.q_targets
            data.positives = data.sim_list[torch.arange(B), targets]
            # Create a mask for negative scores
            mask = torch.ones((B, S), dtype=torch.bool)
            mask[torch.arange(B), targets] = 0

            # Get negative scores using the mask
            data.negatives = data.sim_list[mask].view(B, S - 1)
        return data

    @staticmethod
    def get_topk_values(matrix: Tensor, k: int, dim: int) -> Tensor:
        return torch.topk(matrix, k, dim)[0]

    @staticmethod
    def _geometric_mean(t: Tensor, dim=1, keepdim=True) -> Tensor:
        log_tensor = torch.log(t)
        mean = torch.mean(log_tensor, dim=dim, keepdim=keepdim)
        geom_mean = torch.exp(mean)
        return geom_mean

    def backward(self, *args, **kwargs):
        data = self.data
        self.loss = self.criterion(data.q_F,
                                   data.S_F.view(len(data.S_F) // data.shot_num, data.shot_num, -1)[data.q_targets],
                                   None,
                                   data.qv,
                                   data.positives,
                                   data.negatives)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def calculate_loss(self, pred, gt):
        return self.eval_criterion(pred, gt)
