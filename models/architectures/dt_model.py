import time
from datetime import datetime
from typing import Iterator
from skimage.color import rgb2hsv
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.nn import Parameter
from torch.utils.data import DataLoader as TorchDataLoader, DataLoader

from models import backbones, de_heads, necks
from models.de_heads.dengine import DecisionEngine
from models.de_heads.dtree import DTree
from models.interfaces.arch_module import ARCH
from models.utilities.utils import DataHolder, config_exchange, AverageMeter
from plot_thesis import visualize_embeddings


class CNNModel(ARCH):
    """
    Provides model building functionalities
    Can be used as a generic model as well.
    """
    arch = 'Missing'
    DE: DecisionEngine = None

    def __init__(self, cfg_path):
        super().__init__(cfg_path)
        model_cfg = self.root_cfg
        self.num_classes = model_cfg.WAY_NUM
        self.k_neighbors = model_cfg.K_NEIGHBORS
        self.build()
        self._set_modules_mode()
        self.out_bank = np.empty(shape=(0, self.num_classes))
        self.target_bank = None
        self.loss_tracker = AverageMeter()

    def load_data(self, mode, output_file, dataset_dir=None):
        self.loaders = self.ds_loader.load_data(mode, dataset_dir, output_file)

    def _build_ENCODER(self):  # ENCODER | _
        if self.root_cfg.get("ENCODER", None) is None:
            raise ValueError('Missing specification of encoder to use')
        m = necks.__all__[self.root_cfg.ENCODER.NAME]
        m = m(self.override_child_cfg(m.get_config(), "ENCODER"))
        m.cuda() if self.root_cfg.ENCODER.CUDA else False
        self.module_topology['ENCODER'] = m
        self.data.module_list.append(m)
        return m

    def _build_DE(self):
        if self.root_cfg.get('DE', None) is None:
            return None
        de: DecisionEngine = de_heads.__all__[self.root_cfg.DE.NAME]
        de = de(config_exchange(de.get_config(), self.root_cfg['DE']))
        self.module_topology['DE'] = de
        de.cuda() if self.root_cfg.DE.CUDA else False
        self.DE = de
        return de

    def run_epoch(self, output_file):
        """
        Functionality: Iterate over training dataset episodes , pre-process the query and support classes and update
        the model IO-object (DataHolder). Run inference and collect loss for performance tracking.
        :param output_file: opened txt file for logging
        :type output_file: IOFile
        :return: None
        """
        raise NotImplementedError

    def save_model(self, filename=None):
        if getattr(self, 'DE', None) is not None:
            self.state.update({'DE': self.DE.dump()})
        super().save_model(filename)

    def load_model(self, path, txt_file=None):
        checkpoint = super().load_model(path, txt_file)
        if checkpoint is not None:
            if 'DE' in checkpoint and 'DE' in self.root_cfg:
                self.DE.load(checkpoint['DE'])

    def validate(self, val_loader: DataLoader, best_prec1: float, F_txt , store_output=False, store_target=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        self.eval()
        accuracies = []

        end = time.time()
        self.data.training(False)

        with torch.no_grad():
            for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(
                    val_loader):

                # Convert query and support images
                query_images = torch.cat(query_images, 0)

                input_var1 = query_images.cuda()
                input_var2 = torch.cat(support_images, 0).squeeze(0).cuda()
                input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3),
                                                          input_var2.size(4))
                # Deal with the targets
                target = torch.cat(query_targets, 0).cuda()

                self.data.q_CPU = query_images
                self.data.q_in, self.data.S_in = input_var1, input_var2

                out = self.forward()
                loss = self.calculate_loss(out, target)

                # measure accuracy and record loss
                losses.update(loss.item(), query_images.size(0))

                prec1, _ = self.calculate_accuracy(out, target, topk=(1, 3))

                top1.update(prec1[0], query_images.size(0))
                accuracies.append(prec1)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if isinstance(out, torch.Tensor):
                    out = out.detach().cpu().numpy()
                if store_output:
                    self.out_bank = np.concatenate([self.out_bank, out], axis=0)
                if store_target:
                    self.target_bank = np.concatenate([self.target_bank, target.cpu().numpy()], axis=0)
                # ============== print the intermediate results ==============#
                if episode_index % 100 == 0 and episode_index != 0:
                    print(f'Test-({self.get_epoch() - 1}): [{episode_index}/{len(val_loader)}]\t'
                          f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                          f'Prec@1 {top1.val} ({top1.avg})\t')

                    F_txt.write(f'\nTest-({self.get_epoch() - 1}): [{episode_index}/{len(val_loader)}]\t'
                                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                                f'Prec@1 {top1.val} ({top1.avg})\n')
        self.loss_tracker.loss_list = losses.loss_list
        # self.write_losses_to_file(self.loss_tracker.get_loss_history())
        top1.avg = top1.avg.item() if isinstance(top1.avg, torch.Tensor) else top1.avg
        best_prec1 = best_prec1.item() if isinstance(best_prec1, torch.Tensor) else best_prec1
        print(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}')
        F_txt.write(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}')
        return top1.avg, accuracies
