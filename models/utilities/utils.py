
from __future__ import print_function, annotations

import dataclasses
import functools
import gc
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import yaml
from easydict import EasyDict
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
from torch import Tensor, nn as nn
from torch.nn import BatchNorm2d, init
import seaborn as sns
from torch.nn import functional as F
"""
General utilities
"""


def create_confusion_matrix(true: np.ndarray, pred: np.ndarray):
    cm = confusion_matrix(true, pred)

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax)
    ax.grid(False)
    plt.show()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_config(config: Path, root_cfg=None, ret_dict=False):
    with open(config, mode="r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    if root_cfg is not None:
        config_exchange(cfg, root_cfg)
    if ret_dict:
        return EasyDict(cfg), cfg
    return EasyDict(cfg)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


@dataclass
class DataHolderBase:
    """
     Module IO object
     """

    cfg: Any
    module_list: List[nn.Module]
    k_neighbors: int  # 1 - num_classes
    use_bias: bool
    norm_layer: Any  # torch

    def __init__(self, cfg):
        self.cfg = cfg

    def __post_init__(self):
        cfg = self.cfg
        self.k_neighbors = cfg.K_NEIGHBORS
        self.use_bias = cfg.USE_BIAS
        self.norm_layer = BatchNorm2d


def config_exchange(dest: dict, src: dict):
    for k, v in src.items():
        if k in dest:
            if v is None:
                src[k] = dest[k]
            else:
                dest[k].update(v)
    for k, v in dest.items():
        if k not in src:
            src[k] = v
    print(src)


class DataHolder(DataHolderBase):
    """
    Module IO specification object
    Models can choose to employ it for sharing information explicitly among child modules.
    """
    # Classification
    num_classes: int
    # ======== DNX ==========
    q_CPU: Tensor
    q_in: Tensor
    S_in: Tensor
    q_targets: Tensor
    q_permuted_targets: Tensor
    S_targets: Tensor
    qv: int
    sv: int
    cos_sim: Tensor
    # Backbone2d OUTPUT
    q_F: Tensor
    DLD_topk: Tensor
    S_F: List[Tensor] | Tensor  # CUDA

    # Tree fit input
    X: DataFrame
    y: DataFrame
    eval_set: Tuple[Any, Any]

    # ======== SNX ==========
    snx_queries: Tensor
    positives: Tensor
    negatives: Tensor
    snx_support_sets: Tensor
    # SNX embeddings
    snx_query_f: Tensor
    snx_positive_f: Tensor
    snx_negative_f: Tensor
    snx_support_set_f: Tensor
    # Tree-out
    tree_pred: np.ndarray

    # ======== Output ==========
    sim_list: Tensor  # CUDA
    apn: Tensor  # CUDA
    output: Any  # CUDA

    def __init__(self, cfg):
        super().__init__(cfg)
        self._train = True
        self.eval_set = None
        self.module_list: List = []
        self.__post_init__()

    def __post_init__(self):
        super().__post_init__()
        cfg = self.cfg
        self.num_classes = cfg.WAY_NUM
        self.shot_num = cfg.SHOT_NUM
        self.qv = cfg.AUGMENTOR.QAV_NUM or 0  # query views (per unique query)
        self.sv = cfg.AUGMENTOR.SAV_NUM or 0  # support views (per shot)

    def reset(self):
        self.empty_cache()
        self.__post_init__()

    def empty_cache(self):
        attributes = ['q_in', 'S_in', 'q_F', 'S_F', 'output', 'apn', 'cos_sim', 'sim_list', 'snx_queries',
                      'snx_positives',
                      'snx_support_sets', 'snx_query_f', 'snx_positive_f', 'snx_negative_f', 'snx_support_set_f',
                      'tree_pred', 'DLD_topk']
        for attr in attributes:
            try:
                delattr(self, attr)
            except AttributeError:
                pass
        gc.collect()
        torch.cuda.empty_cache()

    def get_qv(self):
        return self.qv

    def get_Sv(self):
        return self.sv

    def training(self, mode=True):
        self._train = mode

    def is_training(self):
        return self._train


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.loss_list = []
        self.loss_history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.loss_list = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.loss_list.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_loss_history(self):
        self.loss_history.append(np.mean(self.loss_list))
        self.loss_list = []
        return self.loss_history


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape((1, -1)).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape((-1,)).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def net_init_weights_normal(net: nn.Module):
    for m in net.modules():
        if isinstance(m, nn.Conv1d):
            init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m: nn.Module):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)
    elif max([classname.find('ReLU'),
              classname.find('LeakyReLU'),
              classname.find('Softmax'),
              classname.find('Tanh'),
              classname.find('Sigmoid')]) != -1:
        return
    elif isinstance(m, nn.Sequential) or isinstance(m, nn.Module):
        print("Attempting weight initialisation recursively for:", classname)
        for layer in m.children():
            weights_init_kaiming(layer)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    use_bias = norm_layer.func == nn.InstanceNorm2d

    return norm_layer, use_bias


def geometric_mean(t: Tensor, dim=0, keepdim=False) -> Tensor:
    return torch.exp(torch.mean(torch.log(t), dim=dim, keepdim=keepdim))


def identity(x):
    return x


def deep_convert_easydict(layer):
    to_ret = layer
    if isinstance(layer, EasyDict):
        to_ret = dict(layer)

    try:
        for key, value in to_ret.items():
            to_ret[key] = deep_convert_easydict(value)
    except AttributeError:
        pass

    return to_ret





def save_attention_map_as_image(source_images: torch.Tensor, attention_masks: torch.Tensor, save_dir="attention_maps"):
    """
    Save a batch of attention maps as images.

    :param source_images: torch tensor of shape (B, 3, H, W)
    :param attention_masks: torch tensor of shape (B, 1, H/4, W/4)
    :param save_dir: directory to save the attention maps
    """
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Resize the attention masks to match the source image size
    attention_masks = F.interpolate(attention_masks, size=source_images.shape[2:], mode='bilinear', align_corners=False)

    # Normalize the attention masks to range between 0 and 1
    attention_masks = (attention_masks - attention_masks.min()) / (attention_masks.max() - attention_masks.min())

    for idx, (img, mask) in enumerate(zip(source_images, attention_masks)):
        # Convert tensors to numpy arrays
        img = img.permute(1, 2, 0).numpy()
        mask = mask.squeeze().numpy()

        # Use the 'cool' colormap to get colors ranging from blue to red
        color_map = plt.cm.viridis(mask)

        # Overlay the colored mask on the source image using the mask values as alpha values
        overlay = img + (color_map[:, :, :3] - img) * mask[..., np.newaxis]
        overlay = np.clip(overlay, 0, 1)  # Ensure values are within [0, 1]

        # Save the image
        plt.imsave(os.path.join(save_dir, f"attention_map_{idx}.png"), overlay)

