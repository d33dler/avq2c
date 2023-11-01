import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cosine_similarity


class TripletMarginWithDistanceLossImageToClass(nn.Module):
    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super(TripletMarginWithDistanceLossImageToClass, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, distances_batch, target_batch):
        losses = []

        for distances, target in zip(distances_batch, target_batch):
            positive_class = target.item()
            positive_dist = distances[positive_class]

            negative_dists = torch.cat((distances[:positive_class], distances[positive_class + 1:]))
            negative_dist, _ = torch.min(negative_dists, dim=0)

            loss = torch.clamp(positive_dist - negative_dist + self.margin, min=0.0)
            losses.append(loss)

        losses = torch.stack(losses)

        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:
            return losses


class NPairMCLoss(nn.Module):
    """
    Original Multi-class NPair loss
    (K. Sohn. Improved Deep Metric Learning with Multi-class N-pair Loss Objective. NIPS 2016)
    """

    def __init__(self, l2_reg=0.02):  # add a regularization coefficient
        super(NPairMCLoss, self).__init__()
        self.l2_reg = l2_reg

    def _calc_cosine_similarities(self, queries, positives, negatives, av):
        """
        Compute the cosine similarity between query and positive and negatives.

        Parameters
        ----------
        queries : torch.Tensor
            Tensor of query embeddings of shape [batch_size, embedding_dim]
        positives : torch.Tensor
            Tensor of positive embeddings of shape [batch_size // av, embedding_dim]
        negatives : torch.Tensor
            Tensor of negative embeddings of shape [batch_size // av, num_negatives, embedding_dim]
        av : int
            Number of augmented views. If av = 0, then just compute the distance without averaging.

        Returns
        -------
        query_pos_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and positive of shape [batch_size // av,]
        query_neg_cos_sim : torch.Tensor
            Tensor of cosine similarities between query and negatives of shape [batch_size // av, num_negatives]
        """
        batch_size = queries.size(0)

        # Reshape queries, positives, and negatives tensor to compute cosine similarity for each view
        queries = queries.view(batch_size // av, av, -1)
        positives = positives.view(batch_size // av, av, -1)
        negatives = negatives.view(batch_size // av, av, -1, negatives.size(-2), negatives.size(-1))

        # Compute cosine similarity between each view of each query and each view of its corresponding positive
        query_pos_cos_sim = torch.stack(
            [cosine_similarity(queries[:, i, :], positives[:, j, :]) for j in range(av) for i in range(av)])

        # Compute cosine similarity between each view of each query and each view of its corresponding negatives
        query_neg_cos_sim = torch.stack(
            [cosine_similarity(queries[:, i, :].unsqueeze(1), negatives[:, j, :, :, :], dim=-1) for j in range(av) for i
             in range(av)])

        if av > 0:
            # Compute arithmetic mean across the augmented views
            query_pos_cos_sim = torch.mean(query_pos_cos_sim, dim=0)
            query_neg_cos_sim = torch.mean(query_neg_cos_sim, dim=0)

        return query_pos_cos_sim.squeeze(), query_neg_cos_sim.squeeze()

    def forward(self, anchors, positives, negatives, av=1, pos_sim=None, neg_sim=None):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        pos_sim, neg_sim = self._check_calc_csim(anchors, positives, negatives, av, pos_sim, neg_sim)

        loss = torch.log1p(torch.sum(torch.exp(neg_sim - pos_sim.unsqueeze(1)), dim=1)).mean() \
               + self.l2_loss(anchors, positives) * self.l2_reg
        return loss

    def _check_calc_csim(self, anchors, positives, negatives, av=0, pos_sim=None, neg_sim=None):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        if pos_sim is None or neg_sim is None:
            pos_sim, neg_sim = self._calc_cosine_similarities(anchors, positives, negatives, av)
        return pos_sim, neg_sim

    def l2_loss(self, anchors, positives):
        """
        Calculates L2 norm regularization loss on flattened tensors
        :param anchors: A torch.Tensor, (n, embedding_size)
        :param positives: A torch.Tensor, (m, embedding_size)
        :return: A scalar
        """
        total_elements = torch.numel(anchors) + torch.numel(positives)
        return (torch.sum(anchors ** 2) + torch.sum(positives ** 2)) / total_elements


class NPairMCLossLSE(NPairMCLoss):
    """
    LSE version Multi-class NPair loss (w/ Log-Sum-Exp for numerical stability)
    based on (K. Sohn. Improved Deep Metric Learning with Multi-class N-pair Loss Objective. NIPS 2016)
    """

    def __init__(self, l2_reg=0.02):  # add a regularization coefficient
        super(NPairMCLossLSE, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchors, positives, negatives, av=0, pos_sim=None, neg_sim=None):
        """
        positives: 1D tensor of shape (B,)
        negatives: 2D tensor of shape (B,(L-1) * AV)
        """
        # Maximum value for stability
        pos_sim, neg_sim = self._check_calc_csim(anchors, positives, negatives, av, pos_sim, neg_sim)
        with torch.no_grad():
            max_val = torch.max(neg_sim - pos_sim.unsqueeze(1), dim=1, keepdim=True)[0]
        loss = (max_val + torch.log(
            torch.sum(torch.exp(neg_sim - pos_sim.unsqueeze(1) - max_val), dim=1))).mean() \
               + self.l2_loss(anchors, pos_sim) * self.l2_reg
        return loss


class NPairAngularLoss(nn.Module):
    def __init__(self, alpha=45, in_degree=True, with_npair=True):
        super(NPairAngularLoss, self).__init__()
        if in_degree:
            alpha = np.deg2rad(alpha)
        self.npair = None
        if with_npair:
            self.npair = NPairMCLoss()
        self.sq_tan_alpha = np.tan(alpha) ** 2

    def forward(self, anchors, positives, apn, negatives, pos_sim, neg_sim, lamb=2):
        """
        Compute the angular loss for a batch of anchors, positives and negatives
        including (optionally) the npair loss
        :param anchors: (batch_size, embedding_size)
        :param positives: (batch_size, embedding_size)
        :param apn: anchor + positive (batch_size, )
        :param negatives: (batch_size, num_negatives, embedding_size)
        :param pos_sim: (batch_size, )
        :param neg_sim: (batch_size, num_negatives)
        :param lamb: the weight of the angular loss
        :return:

        """
        term1 = 4 * self.sq_tan_alpha * apn
        term2 = 2 * (1 + self.sq_tan_alpha) * pos_sim.unsqueeze(1)
        f_apn = term1 - term2
        with torch.no_grad():
            max_val = torch.max(f_apn, dim=1, keepdim=True)[0]
        loss = torch.mean(max_val + torch.log1p(torch.sum(torch.exp(f_apn - max_val), dim=1)))
        if self.npair is not None:
            loss_npair = self.npair(anchors, positives, negatives, pos_sim=pos_sim, neg_sim=neg_sim)
            # print(loss, loss_npair)
            loss = loss_npair + lamb * loss
        return loss


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int, device=torch.device('cpu'), reg_lambda: float = 1.0,
                 reg_alpha: float = 0.5):
        """
        Arguments:
            num_classes: number of classes.
            feat_dim: feature dimension.
            device: device to run the module.
            reg_lambda: regularization coefficient for the center loss.
            reg_alpha: regularization coefficient for the center update.
        """
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self._criterion = nn.MSELoss()
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha

    def forward(self, features, targets):
        """
        Arguments:
            features: features with shape (batch_size, feat_dim).
            targets: ground truth labels with shape (batch_size).
        """
        # Compute the loss
        target_centers = self.centers[targets]

        center_loss = self._criterion(features, target_centers) * self.reg_lambda

        # Update the centers without updating the gradients
        with torch.no_grad():
            delta = self.get_center_delta(features, self.centers, targets)
            self.centers = nn.Parameter(self.centers - delta)

        return center_loss

    def get_center_delta(self, features, centers, targets):
        """
        Arguments:
            features: features with shape (batch_size, feat_dim).
            centers: centers with shape (num_classes, feat_dim).
            targets: ground truth labels with shape (batch_size).
        """
        targets, indices = torch.sort(targets)
        target_centers = centers[targets]
        features = features[indices]

        delta_centers = target_centers - features
        uni_targets, indices = torch.unique(
            targets, sorted=True, return_inverse=True)

        uni_targets = uni_targets.to(self.device)
        indices = indices.to(self.device)

        delta_centers = torch.zeros(
            uni_targets.size(0), delta_centers.size(1)
        ).to(self.device).index_add_(0, indices, delta_centers)

        targets_repeat_num = uni_targets.size()[0]
        uni_targets_repeat_num = targets.size()[0]
        targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
        uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
        same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

        delta_centers = delta_centers / (same_class_feature_count + 1.0) * self.reg_alpha
        result = torch.zeros_like(centers)
        result[uni_targets, :] = delta_centers
        return result
