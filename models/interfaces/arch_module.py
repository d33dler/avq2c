from __future__ import annotations

import functools
import os
from abc import ABC, abstractmethod
from enum import Enum, EnumMeta
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch import nn, Tensor
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import json
from data_loader.data_load import Parameters, DatasetLoader
from models import backbones
from models.utilities.utils import save_checkpoint, load_config, accuracy, config_exchange, DataHolder


# noinspection PyUnresolvedReferences
class ARCH(nn.Module):
    """
    Module architecture class - wraps around nn.Module and organizes necessary functions/variables and provides necessary
    model functionalities. This class also hosts an inner class ARCH.Child to be implemented by any module used in
    a model - ARCH stores the module instances and offers module management functionalities.
    Functionalities:
    Save & Load model
    Override child module configuration
    (root config child-module sub-configs fields matching child module configuration fields)
    Return calculated loss
    Calculate accuracy
    LR tweaking
    Any model in this framework must subclass Class.ARCH !
    """
    root_cfg: EasyDict
    optimizers: Dict
    best_prec1 = 0
    _epochix = 0
    _loss_val: Tensor
    loaders = None

    class ActivationFuncs(Enum):
        Relu = nn.ReLU
        Lrelu = nn.LeakyReLU
        Sigmoid = nn.Sigmoid

    class NormalizationFuncs(Enum):
        BatchNorm2d = functools.partial(nn.BatchNorm2d, affine=True)
        InstanceNorm2d = functools.partial(nn.InstanceNorm2d, affine=False)
        none = None

    class PoolingFuncs(Enum):
        MaxPool2d = nn.MaxPool2d
        AveragePool2d = nn.AvgPool2d
        LPPool2d = nn.LPPool2d

    class Child(nn.Module, ABC):
        """
        ARCH.Child class - abstract class wrapping around nn.Module and providing typical
        module training functionalities. Any model sub-classing ARCH should employ modules
        sub-classing ARCH.Child.
        Functionalities:
        Loss calculation
        Backward call
        LR adjustment
        """
        config: Dict | EasyDict
        lr: float
        optimizer: Optimizer
        criterion: nn.Module | List[nn.Module]
        scheduler: _LRScheduler
        loss: Tensor
        fine_tuning = True
        freeze_epoch = 1

        def __init__(self, config: EasyDict | dict) -> None:
            super().__init__()
            self.config = config
            self.scheduler = None

        def calculate_loss(self, *args):
            """
            Calculates without calculating the gradient
            :param gt: ground truth
            :type gt: Sequence
            :param pred: predictions
            :type pred: Sequence
            :return: loss
            :rtype: Any
            """
            return self.criterion(*args) if isinstance(self.criterion, nn.Module) else self.criterion[0](*args)

        def backward(self, *args, **kwargs):
            """
            Calculates the gradient and runs the model DAG backward
            Default implementation assumes args are (pred, gt)
            :param args: arguments
            :type args: Sequence
            :param kwargs: keyword arguments
            :type kwargs: Dict
            :return: loss
            :rtype: Any
            """
            pred, gt = args
            self.loss = self.criterion(pred, gt) if isinstance(self.criterion, nn.Module) else sum(
                [criterion(pred, gt) for criterion in self.criterion])
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            return self.loss

        def adjust_learning_rate(self, epoch):
            lr = self.lr * (0.5 ** (epoch // 10))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        def freeze_layers(self):
            pass

        def set_optimize(self, optimize=True):
            self.fine_tuning = optimize

        def load_optimizer_state_dict(self, optim_state_dict):
            self.optimizer.load_state_dict(optim_state_dict)

        def load_scheduler_state_dict(self, scheduler_state_dict):
            self.scheduler.load_state_dict(scheduler_state_dict)

        @staticmethod
        @abstractmethod
        def get_config():
            return None

        def load(self, model: Any) -> None:
            pass

        def dump(self) -> Any:
            return None

    class BaseConfig:
        __doc__ = "Base config class for yaml files. Override __doc__ for implementations."

    module_topology: Dict[str, Child] = None
    ds_loader: DatasetLoader = None

    def __init__(self, cfg_path) -> None:
        super().__init__()
        self.root_cfg_dict = None
        self.cfg_path = cfg_path
        self._store_path = None
        self.state = dict()
        self.load_config(cfg_path)
        self.module_topology: Dict[str, ARCH.Child] = self.root_cfg.TOPOLOGY
        self._mod_topo_private = self.module_topology.copy()
        self._freeze_epoch = self.root_cfg.BACKBONE.FREEZE_EPOCH

        c = self.root_cfg
        p = Parameters(c.SHOT_NUM, c.WAY_NUM, c.QUERY_NUM, c.EPISODE_TRAIN_NUM,
                       c.EPISODE_TEST_NUM, c.EPISODE_VAL_NUM, c.OUTF, c.WORKERS, c.EPISODE_SIZE,
                       c.TEST_EPISODE_SIZE, c.QUERY_NUM * c.WAY_NUM, c.get("BUILDER_TYPE", None))
        self.data = DataHolder(c)
        self.dataset_parameters = p
        if self.ds_loader is None:
            self.ds_loader = DatasetLoader(self.data, p)

    def forward(self):
        raise NotImplementedError

    def run_epoch(self, output_file):
        raise NotImplementedError

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def build(self):
        for module_name in self.module_topology.keys():
            module = getattr(self, '_build_%s' % module_name)()
            self.add_module(module_name, module)

    def _build_BACKBONE(self):
        if self.root_cfg.get("BACKBONE", None) is None:
            raise ValueError('Missing specification of backbone to use')
        m: ARCH.Child = backbones.__all__[self.root_cfg.BACKBONE.NAME](self.data)
        m.cuda() if self.root_cfg.BACKBONE.CUDA else False  # TODO may yield err?
        self.module_topology['BACKBONE'] = m
        self.data.module_list.append(m)
        return m

    def verify_module(self, module):
        if not isinstance(module, ARCH.Child):
            raise ValueError(
                "[CFG_OVERRIDE] Cannot override child module config. Child module doesn't subclass ARCH.Child!")

    def backward(self, *args, **kwargs):
        loss_ls = []
        for m in self.module_topology.values():
            loss_ls.append(m.backward(*args, **kwargs))
        return loss_ls

    @staticmethod
    def get_func(fset: EnumMeta, name: str):
        if name not in fset.__members__.keys():
            raise NotImplementedError('Function [%s] not found' % str)
        return fset[name].value

    def _set_modules_mode(self):
        for k, m in self.module_topology.items():
            v = self.root_cfg[k].MODE == 'TRAIN'
            print("Setting module: ", k, " TRAIN" if v else " TEST", " mode.")
            m.train(v)

    def save_model(self, filename=None):
        """
        Saves model to filename or back to same checkpoint file loaded into the model.
        Subclasses can store in the ARCH.state field additional components  and call this function
        to save everything.
        :param filename:
        :type filename:
        :return:
        :rtype:
        """
        # TODO add root cfg saving to state and version+parameter cross-checking
        if filename is None:
            filename = self._store_path
            if self._store_path is None:
                raise ValueError("Missing model save path!")

        priv = self._mod_topo_private

        state = {
            'epoch_index': self._epochix,
            'arch': self.arch,
            'best_prec1': self.best_prec1,

        }
        optimizers = {
            f"{priv[k]}_optim": v.optimizer.state_dict() for k, v in self.module_topology.items() if
            v.optimizer is not None}
        schedulers = {
            f"{priv[k]}_scheduler": v.scheduler.state_dict() for k, v in self.module_topology.items() if
            v.scheduler is not None
        }
        state_dicts = {f"{priv[k]}_state_dict": v.state_dict() for k, v in self.module_topology.items()}

        state.update(optimizers)
        state.update(state_dicts)
        state.update(schedulers)
        state.update(self.state)
        save_checkpoint(state, filename)
        print("Saved model to:", filename)
        # save config
        with open(self.cfg_path, 'w') as f:
            self.root_cfg_dict['RESUME'] = filename
            yaml.dump(self.root_cfg_dict, f, default_flow_style=False)

        print("Saved config file:", self.cfg_path)

    def load_model(self, path, txt_file=None):
        """
        Load models main components from checkpoint : modules state-dictionaries & optimizers-state-dictionaries
        Method returns the checkpoint for any subclass of ARCH to load additional (model specific) components.
        :param path: file path
        :type path: Path | str
        :param txt_file: logging file
        :type txt_file: IOFile
        :return: torch.checkpoint | None
        :rtype: Any
        """
        priv = self._mod_topo_private
        self._store_path = path
        if os.path.isfile(path):
            print("=> loading checkpoint '{}'".format(path))
            checkpoint = torch.load(path)
            self._epochix = checkpoint['epoch_index']
            self.best_prec1 = checkpoint['best_prec1']
            [v.load_state_dict(checkpoint[f"{priv[k]}_state_dict"]) for k, v in self.module_topology.items()
             if f"{priv[k]}_state_dict" in checkpoint]
            [v.load_optimizer_state_dict(checkpoint[f"{priv[k]}_optim"]) for k, v in self.module_topology.items()
             if f"{priv[k]}_optim" in checkpoint]
            [v.load_scheduler_state_dict(checkpoint[f"{priv[k]}_scheduler"]) for k, v in self.module_topology.items()
             if f"{priv[k]}_scheduler" in checkpoint]

            print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch_index']))
            if txt_file:
                print("=> loaded checkpoint '{}' (epoch {})".format(path, checkpoint['epoch_index']), file=txt_file)
            return checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(path))
            if txt_file:
                print("=> no checkpoint found at '{}'".format(path), file=txt_file)
            return None

    def load_config(self, path):
        self.root_cfg, self.root_cfg_dict = load_config(path, ret_dict=True)
        # pretty print config
        print("Loaded config file:", path)
        print("Config:")
        print(json.dumps(self.root_cfg_dict, indent=4))

    def override_child_cfg(self, _config: EasyDict | dict, module_id: str):
        """
        Overrides ARCH.Child module configuration based on root config values.
        Important:
        The mappings should be nested in the root config in the module's YAML objects (e.g. BACKBONE, DT...)
        and should not be nested in the child config!
        The keys in child config and root config must match (obviously).

        :param _config: child config
        :param module_id: child module ID
        :return: child cfg: EasyDict | dict
        """
        if module_id not in self.root_cfg.keys():
            raise KeyError("[CFG_OVERRIDE] Module ID not found in root cfg!")
        return config_exchange(_config, self.root_cfg[module_id])

    def get_loss(self, module_id: str = None):
        if module_id is None:
            return self.module_topology[self.root_cfg.TRACK_LOSS].loss
        if module_id not in self.module_topology.keys():
            raise KeyError(f"Module {module_id} not found in architecture topology")
        return self.module_topology[module_id].loss

    def calculate_loss(self, *args):
        """
        Calculate loss (without grad!)
        """
        return self.module_topology[self.root_cfg.TRACK_LOSS].calculate_loss(*args)

    def calculate_accuracy(self, output, target, topk=(1,)):
        prec = accuracy(F.softmax(output, dim=1), target, topk=topk)
        self.best_prec1 = prec[0] if prec[0] > self.best_prec1 else self.best_prec1
        return prec

    def get_epoch(self):
        return self._epochix

    def incr_epoch(self):
        self._epochix += 1

    def get_criterion(self, module_id: str = None):
        if module_id is None or module_id not in self.module_topology.keys():
            return self.module_topology[self.root_cfg.TRACK_CRITERION].criterion
        return self.module_topology[module_id].criterion

    def adjust_learning_rate(self):
        """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
        for k, mod in self.module_topology.items():
            print("Adjusting learning rate for module: ", k)
            mod.adjust_learning_rate(epoch=self._epochix)

    def freeze_auxiliary(self):
        """
        Freeze auxiliary layers and modules (e.g. Dropout, BatchNorm, etc.)
        """
        if self._epochix >= self._freeze_epoch:
            print(" => Freezing auxiliary layers")
            self.eval()
