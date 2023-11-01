import time
from pathlib import Path

import torch

from models.architectures.dt_model import CNNModel
from models.utilities.data_augmentation import read_batch_transform
from models.utilities.utils import AverageMeter, accuracy


class DN_X(CNNModel):
    """
    DN_X (4|7|X) Model
    Implements epoch run employing few-shot learning & performance tracking during training
    Structure:
    [Deep learning module] ⟶ [K-NN module] ⟶ [Decision Tree]

    """
    arch = 'DN4'

    def __init__(self, cfg_path):
        """
        Pass configuration file path to the superclass
        :param cfg_path: model root configuration file path to be loaded (should include sub-module specifications)
        :type cfg_path: Path | str
        """
        super().__init__(cfg_path)
        self.batch_augment = read_batch_transform(self.root_cfg.get("BATCH_AUGMENT", None))

    def forward(self):
        self.BACKBONE.forward()
        return self.data.sim_list

    def run_epoch(self, output_file):
        """
        Run a training epoch over the dataset, applying batch augmentations, calculating loss, and updating the model.

        This method iterates over episodes in the training dataset, preprocesses the query and support sets, and updates
         the DataHolder object. It then performs inference, calculates the loss, and logs performance metrics.

        CutMix is a data augmentation technique applied during training, where parts of images and their corresponding
        labels are cut and mixed from different training examples. If batch augmentation is enabled and
        CutMix is specified, it is applied to the query images in each episode.
        Next iteration - CutMix will be moved to an outside routine.

        Parameters:
        output_file (IOFile): A text file open for writing, used to log the training progress and performance metrics.

        Returns:
        None

        The method updates internal state related to training progress and logs the epoch's performance metrics to the
         provided output file. It also handles any necessary learning rate adjustments and epoch incrementation.
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        train_loader = self.loaders.train_loader
        end = time.time()
        epochix = self.get_epoch()

        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(
                train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)

            # Convert query and support images
            query_images = torch.cat(query_images, 0)


            input_var2 = torch.cat(support_images, 0).squeeze(0).cuda()
            input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))
            # Deal with the targets
            target = torch.cat(query_targets, 0).cuda()

            query_num = self.data.cfg.QUERY_NUM * self.data.qv
            B = query_images.size(0)

            if self.batch_augment is not None:
                permuted_targets = torch.zeros((len(target), 3), dtype=torch.float).cuda()
                target_indices = torch.arange(0, self.data.cfg.WAY_NUM, dtype=torch.float).cuda()
                for i in range(0, query_num, self.data.qv):
                    cutmix_indices = torch.arange(i, B, query_num)
                    cutmix_query_images = query_images[cutmix_indices]
                    query_images[cutmix_indices], permuted_targets[cutmix_indices // self.data.qv] = \
                        self.batch_augment(cutmix_query_images, target_indices)
                self.data.q_permuted_targets = permuted_targets

            else:
                self.data.q_permuted_targets = None

            input_var1 = query_images.cuda()

            self.data.q_targets = target
            self.data.S_targets = torch.cat(support_targets, 0).cuda()
            self.data.q_CPU = query_images
            self.data.q_in = input_var1
            self.data.S_in = input_var2

            # Calculate the output
            out = self.forward()
            loss = self.backward(out, target)[0]
            # Measure accuracy and record loss
            prec1, _ = accuracy(out, target, topk=(1, 3))

            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0], target.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.data.empty_cache()
            # ============== print the intermediate results ==============#
            if episode_index % self.root_cfg.PRINT_FREQ == 0 and episode_index != 0:
                print(f'Eposide-({epochix}): [{episode_index}/{len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      )

                print('Eposide-({0}): [{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epochix, episode_index, len(train_loader),
                                                                      batch_time=batch_time, data_time=data_time,
                                                                      loss=losses,
                                                                      top1=top1), file=output_file)
        self.incr_epoch()
        self.adjust_learning_rate()
