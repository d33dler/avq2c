from dataclasses import field

import torch
import torch.nn as nn
from torch import Tensor

from models.backbones.base2d import BaseBackbone2d
from models.clustering.dn4_nbnn import I2C_KNN
from models.utilities.custom_loss import CenterLoss
from models.utilities.utils import DataHolder, get_norm_layer, net_init_weights_normal


##############################################################################
# Class: FourLayer_64F
##############################################################################

# Model: FourLayer_64F
# Input: One query image and a support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class BaselineBackbone2d(BaseBackbone2d):
    class Config(BaseBackbone2d.RemoteYamlConfig):
        FILE_PATH = __file__  # mandatory
        FILE_TYPE: str = "YAML"  # mandatory
        NUM_CLASSES: int = field(default_factory=int)  # 5 (commented out = default vals)

    def __init__(self, data: DataHolder):
        super().__init__(self.Config())

        self.data = data
        model_cfg = data.cfg.BACKBONE

        self.require_grad = model_cfg.GRAD

        norm_layer, use_bias = get_norm_layer(model_cfg.NORM)
        self.norm_layer = norm_layer
        self.output_channels = 64
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
        )
        self.knn = I2C_KNN(data.k_neighbors)

        self.FREEZE_LAYERS = [(self.features, [1, 5, 9, 12])]
        self.FREEZE_EPOCH = model_cfg.FREEZE_EPOCH
        self.lr = model_cfg.LEARNING_RATE
        net_init_weights_normal(self)
        self.init_optimizer(model_cfg.OPTIMIZER, epochs=data.cfg.EPOCHS)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.reg = CenterLoss(data.num_classes, 64 * 21 * 21, torch.device('cuda'), reg_lambda=0.1,
                              reg_alpha=0.3).cuda()

    def forward(self):
        data = self.data
        data.q_F = self.features(data.q_in)
        data.S_F = self.features(data.S_in)
        qav_num, sav_num = (data.get_qv(), data.get_Sv())   if data.is_training() else (1, 1)
        data.sim_list = self.knn.forward(data.q_F, data.S_F, qav_num, sav_num,
                                         data.cfg.AUGMENTOR.STRATEGY if data.training else None,
                                         data.cfg.SHOT_NUM)
        self.data.output = data.sim_list
        return data

    def backward(self, *args, **kwargs):
        """
        Calculates the gradient and runs the model DAG backward
        Default implementation assumes args are (pred, gt)
        Contains additional logic for permuted targets (e.g. for CutMix)
        :param args: arguments
        :type args: Sequence
        :param kwargs: keyword arguments
        :type kwargs: Dict
        :return: loss
        :rtype: Any
        """
        pred, gt = args
        data = self.data

        if data.q_permuted_targets is not None:
            main_loss = self.criterion(pred, gt)
            permuted_gt = data.q_permuted_targets[:, 1].long()
            lambda_val: Tensor = data.q_permuted_targets[:, 2]
            if data.qv > 1:  # assumes usage of geometric mean
                lambda_val = lambda_val.pow(1 / data.qv)
            permuted_loss = self.criterion(pred, permuted_gt)
            main_loss = lambda_val * main_loss + (1 - lambda_val) * permuted_loss
            main_loss = main_loss.mean()
        else:
            main_loss = self.criterion(pred, gt)
        # s_targets = torch.cat( [torch.full((data.S_F.size(0) // data.num_classes,), class_idx) for class_idx in
        # data.S_targets.unique(sorted=True)]) center_loss = self.reg(data.S_F.view(data.S_F.size(0), -1),
        # s_targets.cuda())
        self.loss = main_loss  # + center_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def adjust_learning_rate(self, epoch):
        self.scheduler.step()

    def calculate_loss(self, *args):
        pred, gt = args
        return self.criterion(pred, gt)
