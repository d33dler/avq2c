from __future__ import annotations

import dataclasses
from typing import Union

from kornia.augmentation import RandomJigsaw, RandomGaussianNoise
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_loader.datasets_csv import BatchFactory
from models.utilities.utils import DataHolder

"""
Augmentation functions mappings that involve both simple & complex image transformations
"""
TRANSFORM_MAP = {
    "IDENTITY": torch.nn.Identity,
    "RESIZE": transforms.Resize,
    "RANDOM_CROP": transforms.RandomCrop,
    "RANDOM_HORIZONTAL_FLIP": transforms.RandomHorizontalFlip,
    "RANDOM_VERTICAL_FLIP": transforms.RandomVerticalFlip,
    "RANDOM_RESIZED_CROP": transforms.RandomResizedCrop,
    "CENTER_CROP": transforms.CenterCrop,
    "COLOR_JITTER": transforms.ColorJitter,
    "RANDOM_ROTATION": transforms.RandomRotation,
    "RANDOM_ERASING": transforms.RandomErasing,
    "RANDOM_PERSPECTIVE": transforms.RandomPerspective,
    "RANDOM_EQUALIZE": transforms.RandomEqualize,
    "RANDOM_ADJUST_SHARPNESS": transforms.RandomAdjustSharpness,
    "RANDOM_GAUSSIAN_NOISE": RandomGaussianNoise,
    "RANDOM_GAUSSIAN_BLUR": transforms.GaussianBlur,
    "TO_TENSOR": transforms.ToTensor,
    "NORMALIZE": transforms.Normalize,
    "GAUSSIAN_BLUR": transforms.GaussianBlur,
    "AUTO_AUGMENT": transforms.AutoAugment,
    "RANDOM_AUGMENT": transforms.RandAugment,
    "TRIVIAL_AUGMENT": transforms.TrivialAugmentWide,
    "JIGSAW": RandomJigsaw,
}


@dataclasses.dataclass
class Parameters:
    """
    DataLoader parameters (extracted from model root config @see models/architectures/configs)
    """
    shot_num: int
    way_num: int
    query_num: int
    episode_train_num: int
    episode_test_num: int
    episode_val_num: int
    outf: str
    workers: int
    episodeSize: int
    test_ep_size: int
    batch_sz: int
    builder_type: str


@dataclasses.dataclass
class Loaders:
    train_loader: Union[DataLoader, None]
    val_loader: Union[DataLoader, None]
    test_loader: DataLoader


class DatasetLoader:
    """
    Class used for data preparation (episode construction, pre-processing, augmentation)
    """

    def __init__(self, data: DataHolder, params: Parameters) -> None:
        self.data = data
        self.params = params
        self.cfg = data.cfg

    def _read_transforms(self, cfg, cfg_aug):
        q_transform_ls, s_transforms = [], []
        for TF in cfg_aug:
            if TF.NAME in cfg.DISABLE:
                continue
            _t = TRANSFORM_MAP[TF.NAME]
            if TF.ARGS:
                if isinstance(TF.ARGS, dict):
                    _t = _t(**TF.ARGS)
                else:
                    _t = _t(*tuple(TF.ARGS))
            else:
                _t = _t()
            if TF.get('Q', None) is not None:
                if TF.Q:
                    q_transform_ls.append(_t)
                if TF.S:
                    s_transforms.append(_t)
            else:
                q_transform_ls.append(_t)
                s_transforms.append(_t)
        return q_transform_ls, s_transforms

    def load_data(self, mode, dataset_directory, F_txt):
        """
        Load and preprocess data for training, validation, and testing.

        This method prepares the datasets for the different modes of operation (training, validation, and testing),
        applies the necessary pre-processing and augmentation, and then wraps them in DataLoader objects for
        efficient batch loading.

        Parameters:
        -----------
        mode : str
            The mode of operation = 'train'|'val'|'test'|. Determines whether to load training, validation and test sets
            or just the test set.
        dataset_directory : str
            The path to the directory containing the dataset.
        F_txt : file object
            An open file object to which log messages are written.

        Returns:
        --------
        Loaders : namedtuple
            A namedtuple containing the DataLoaders for the training, validation, and test sets. The training and
            validation DataLoaders are None when mode is 'test'.

        Notes:
        ------
        - The method uses the parameters defined in self.params and the augmentation configurations defined in
          self.cfg.AUGMENTOR to configure the data loading and augmentation.
        - The method resets the data container (self.data) at the beginning.
        - The BatchFactory class is used to create the datasets, and it takes several parameters including data directory,
          mode, various pre-processing and augmentation options, episode and batch configuration, etc.
        - DataLoader objects are created for the datasets to facilitate batch loading during training or evaluation.
        - The method prints the sizes of the training, validation, and test sets to both the console and the provided file
          object.

        Raises:
        -------
        - Any exceptions raised by the BatchFactory constructor, DataLoader constructor, or other called methods.
        """
        # Method implementation...

        self.data.reset()
        # ======================================= Folder of Datasets =======================================
        # image transform & normalization
        dataset_dir = dataset_directory
        shot_num = self.params.shot_num
        way_num = self.params.way_num
        query_num = self.params.query_num
        episode_train_num = self.params.episode_train_num
        episode_val_num = self.params.episode_val_num
        episode_test_num = self.params.episode_test_num

        cfg_aug = self.cfg.AUGMENTOR
        data = self.data

        pre_process, _ = self._read_transforms(cfg_aug, cfg_aug.PRE_PROCESS)
        Q_augmentation, S_augmentation = self._read_transforms(cfg_aug, cfg_aug.AUGMENTATION)
        post_process, _ = self._read_transforms(cfg_aug, cfg_aug.POST_PROCESS)
        aug_num = cfg_aug.AUG_NUM
        strategy = cfg_aug.STRATEGY
        eval_pre_process = [
            transforms.Resize(92),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        if mode == 'train':
            trainset = BatchFactory(
                builder=self.params.builder_type,
                data_dir=dataset_dir, mode='train',
                pre_process=pre_process,
                Q_augmentations=Q_augmentation,
                S_augmentations=S_augmentation,
                post_process=post_process,
                episode_num=episode_train_num, way_num=way_num, shot_num=shot_num, query_num=query_num,  # batching
                qav_num=data.qv, sav_num=data.sv, aug_num=aug_num, strategy=strategy, use_identity=cfg_aug.USE_IDENTITY,
                is_random_aug=cfg_aug.RANDOM_AUGMENT)  # augmentation
            data.qv = trainset.qv
            data.sv = trainset.sv
            valset = BatchFactory(
                builder=self.params.builder_type,
                data_dir=dataset_dir, mode='val',
                pre_process=eval_pre_process,
                episode_num=episode_val_num, way_num=5, shot_num=shot_num, query_num=query_num)
        testset = BatchFactory(
            builder=self.params.builder_type,
            data_dir=dataset_dir, mode='test',
            pre_process=eval_pre_process,
            episode_num=episode_test_num, way_num=5, shot_num=shot_num, query_num=query_num)
        if mode == 'train':
            print('Trainset: %d' % len(trainset))
            print('Trainset: %d' % len(trainset), file=F_txt)
            print('Valset: %d' % len(valset))
            print('Valset: %d' % len(valset), file=F_txt)
        print('Testset: %d' % len(testset))
        print('Testset: %d' % len(testset), file=F_txt)

        # ========================================== Load Datasets =========================================
        workers = self.params.workers
        train_loader = None
        val_loader = None
        if mode == 'train':
            train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=self.params.episodeSize, shuffle=True,
                num_workers=int(workers), drop_last=True, pin_memory=True
            )
            val_loader = torch.utils.data.DataLoader(
                valset, batch_size=self.params.test_ep_size, shuffle=True,
                num_workers=int(workers), drop_last=True, pin_memory=True
            )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=self.params.test_ep_size, shuffle=True,
            num_workers=int(workers), drop_last=True, pin_memory=True
        )
        return Loaders(train_loader, val_loader, test_loader)
