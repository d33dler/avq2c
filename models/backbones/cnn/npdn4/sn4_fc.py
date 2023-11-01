import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from torch.nn.functional import cosine_similarity

from models.backbones.cnn.dn4.dn4_cnn import BaselineBackbone2d
from models.utilities.custom_loss import NPairMCLoss, NPairAngularLoss
from models.utilities.utils import DataHolder, weights_init_kaiming

"""
NOTE: Not used in the paper
"""
##############################################################################
# Class: SN4_FC
##############################################################################

# Model: Siamese_FourLayer_64F
# Input: Query set, Support set
# Base_model: 4 Convolutional layers --> Image-to-Class layer
# Dataset: 3 x 84 x 84 (miniImageNet & Stanford Dogs)
# Filters: 64->64->64->64
# Mapping Sizes: 84->42->21->21->21

class B_4L64F_MCNP(BaselineBackbone2d):
    """
    Multi-class N-Pair Loss (MCNP)
    """
    def __init__(self, data: DataHolder, config: EasyDict = None):
        super().__init__(data)
        self.data = data
        model_cfg = data.cfg.BACKBONE

        self.require_grad = model_cfg.GRAD

        # norm_layer, use_bias = get_norm_layer(model_cfg.NORM)
        self.output_shape = 1024
        self.fc = nn.Sequential(nn.Flatten(start_dim=1),
                                nn.Linear(64 * 21 * 21, self.output_shape))
        # freeze batchnorm layers
        self.FREEZE_LAYERS = [(self.features, [1, 5, 9, 12])]  # , (self.fc, [1, 4])]
        self.lr = model_cfg.LEARNING_RATE
        # self.features.apply(init_weights_kaiming)
        self.fc.apply(weights_init_kaiming)
        self.criterion = NPairAngularLoss()
        del self.reg

    def forward(self):
        data = self.data
        queries = data.q_in
        data.q_F = F.normalize(self.fc(self.features(queries)), p=2, dim=1)
        support_sets = data.S_in
        data.S_F = F.normalize(self.fc(self.features(support_sets)), p=2, dim=1)
        data.sim_list = self._calc_cosine_similarities_support(data.q_F, data.S_F)
        B, S = data.sim_list.shape
        # sort similarities by positive and negative from data.sim_list
        # Normalize the batch tensor column-wise so that the scores are within [-1, 1]
        data.sim_list = F.softmax(data.sim_list, dim=1)
        sim_list = data.sim_list
        # Assuming you have a targets 1-d vector that specifies the index of the positive class
        targets = data.q_targets
        data.positives = sim_list[torch.arange(B), targets]
        # Create a mask for negative scores
        mask = torch.ones((B, S), dtype=torch.bool)
        mask[torch.arange(B), targets] = 0

        # Get negative scores using the mask
        data.negatives = sim_list[mask].view(B, S - 1)
        return data.sim_list

    def _calc_cosine_similarities_support(self, queries, support_sets):
        """
        Compute the cosine similarity between each query and each sample of each support class.
        Compute geometric means for each query and the support class.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of query embeddings of shape [batch_size, embedding_dim]
        support_sets : torch.Tensor
            Tensor of support sets of shape [num_classes, num_samples_per_class, embedding_dim]

        Returns
        -------
        class_cos_sim : torch.Tensor
            Tensor of cosine similarities between each query and each support class of shape [batch_size, num_classes]
        """
        # Compute cosine similarity between each query and each sample of each support class
        class_cos_sim = cosine_similarity(queries.unsqueeze(1).unsqueeze(1), support_sets.unsqueeze(0), dim=-1)

        # Compute arithmetic mean for each query and the support class
        class_cos_sim = torch.mean(class_cos_sim, dim=2)
        return class_cos_sim

    def backward(self, *args, **kwargs):
        data = self.data
        self.loss = self.criterion(data.q_F,
                                   data.S_F.view(len(data.S_F) // data.shot_num, data.shot_num, -1)[data.q_targets],
                                   data.apn,
                                   None,
                                   data.positives,
                                   data.negatives)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def calculate_loss(self, pred, gt):
        return torch.Tensor([0.0])
