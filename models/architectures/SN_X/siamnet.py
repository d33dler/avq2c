import time
from pathlib import Path

import numpy as np
import torch

from models.architectures.dt_model import CNNModel
from models.utilities.utils import AverageMeter, accuracy


class SN_X(CNNModel):
    """
    Siamese Network (4|7|X) Model
    Implements epoch run employing few-shot learning & performance tracking during training
    Structure:
    [Deep learning module] ⟶ [K-NN module] ⟶ [Decision Tree]

    """
    arch = 'SNX'

    def __init__(self, cfg_path):
        """
        Pass configuration file path to the superclass
        :param cfg_path: model root configuration file path to be loaded (should include sub-module specifications)
        :type cfg_path: Path | str
        """
        super().__init__(cfg_path)

    def forward(self):
        self.BACKBONE.forward()
        return self.data.sim_list

    def run_epoch(self, output_file):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        train_loader = self.loaders.train_loader
        end = time.time()
        epochix = self.get_epoch()

        for episode_index, (queries, positives, targets) in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            # Convert query, positives and targets to cuda tensors
            positives = torch.cat(positives, 0).cuda()
            queries = torch.cat(queries, 0).cuda()
            targets = torch.cat(targets, 0).cuda()

            self.data.q_targets = targets
            self.data.snx_queries = queries
            self.data.positives = positives
            # Calculate the output
            self.forward()
            loss = self.backward()
            # record loss
            n = queries.size(0) // self.data.qv if self.data.qv not in [None, 0] else queries.size(0)
            losses.update(loss.item(), n)

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.data.empty_cache()
            # ============== print the intermediate results ==============#
            if episode_index % self.root_cfg.PRINT_FREQ == 0 and episode_index != 0:
                print(f'Eposide-({epochix}): [{episode_index}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t')
                print('Eposide-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(epochix, episode_index, len(train_loader),
                                                                    batch_time=batch_time, data_time=data_time,
                                                                    loss=losses, ), file=output_file)
        self.incr_epoch()

    def validate(self, val_loader, best_prec1, F_txt, store_output=False, store_target=False):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        # switch to evaluate mode
        self.eval()
        accuracies = []

        end = time.time()
        self.data.training(False)
        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):

            # Convert query and support images
            query_images = torch.cat(query_images, 0)
            input_var1 = query_images.cuda()

            input_var2 = torch.cat(support_images, 0).squeeze(0).cuda().contiguous()
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
                print(f'Test-({self.get_epoch()}): [{episode_index}/{len(val_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val} ({top1.avg})\t')

                F_txt.write(f'\nTest-({self.get_epoch()}): [{episode_index}/{len(val_loader)}]\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                            f'Prec@1 {top1.val} ({top1.avg})\n')
            self.data.empty_cache()
        self.loss_tracker.loss_list = losses.loss_list
        # self.write_losses_to_file(self.loss_tracker.get_loss_history())
        print(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}')
        F_txt.write(f' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}')

        return top1.avg, accuracies

    def backward(self):
        return self.BACKBONE.backward()
