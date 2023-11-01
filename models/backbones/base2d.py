import json
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Tuple

from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from yamldataclassconfig import YamlDataClassConfig, create_file_path_field
from dataclasses_json import DataClassJsonMixin
from models.interfaces.arch_module import ARCH


class BaseBackbone2d(ARCH.Child):
    """
    Abstraction class for 2d backbones implementing using torch
    Layer construction routines are not finished/tested (the rest of the functionalities are used in experiments)
    """
    features: nn.Sequential
    FREEZE_LAYERS: List[Tuple[nn.Module, List[int]]] = []
    ACTIVATION_F: ARCH.ActivationFuncs
    NORMALIZATION_F: ARCH.NormalizationFuncs
    POOLING_F: ARCH.PoolingFuncs
    output_architecture = True

    @dataclass
    class _CFG(ARCH.BaseConfig):

        INP_CHANNELS: List[int] = field(default_factory=list)
        OUT_CHANNELS: List[int] = field(default_factory=list)
        LAYER_STRIDES: List[int] = field(default_factory=list)
        KERNEL_SIZES: List[int] = field(default_factory=list)
        LAYER_POOLS: List[int] = field(default_factory=list)
        LAYER_PADDINGS: List[int] = field(default_factory=list)
        NUM_FILTERS: List[int] = field(default_factory=list)
        MOMENTUM: List[float] = field(default_factory=list)

        UPSAMPLE_STRIDES: List[int] = field(default_factory=list)  # []
        NUM_UPSAMPLE_FILTERS: List[int] = field(default_factory=list)  # []

        ACTIVATION: str = 'Relu'
        NORMALIZATION: str = 'BatchNorm2d'
        POOLING: str = 'MaxPool2d'

        NORM_ARGS: dict = field(default_factory=dict)  # for now only one set of parameters
        POOL_ARGS: dict = field(default_factory=dict)
        DEBLOCK_ARGS: dict = field(default_factory=dict)
        FILE_TYPE: str = "YAML"

        def load_cfg(self, *args, **kwargs):
            raise NotImplementedError("Using base class load(). Must call() implementation instance's function.")

    class RemoteYamlConfig(_CFG, YamlDataClassConfig, ABC):
        pass

    class RemoteJsonConfig(_CFG, DataClassJsonMixin, ABC):
        pass

    def _YamlCFG(self, config: RemoteYamlConfig):
        """Yaml mapping config object class using YamlDataClassConfig."""
        fp = config.FILE_PATH
        config.FILE_PATH = create_file_path_field(Path(
            fp).parent / 'config.yaml')  # os.path.join(os.path.dirname(os.path.realpath(__file__)), Path(load_cfg))
        print(Path(fp).parent / 'config.yaml')
        config.load(Path(fp).parent / 'config.yaml')  # check!
        return config

    def _JsonCFG(self, config: RemoteJsonConfig):
        """Yaml mapping config object class using YamlDataClassConfig."""
        config.FILE_PATH = create_file_path_field(Path(config.FILE_PATH).parent.parent / 'config.json')
        with open(config.FILE_PATH, 'r') as f:
            return self.from_json(json.load(f))

    config: Union[RemoteJsonConfig, RemoteYamlConfig]
    blocks: nn.ModuleList
    deblocks: nn.ModuleList

    def collect_funcs(self):
        self.ACTIVATION_F, \
            self.NORMALIZATION_F, \
            self.POOLING_F = \
            [ARCH.get_func(fset, name)
             for fset, name in [(ARCH.ActivationFuncs, self.config.ACTIVATION),
                                (ARCH.NormalizationFuncs, self.config.NORMALIZATION),
                                (ARCH.PoolingFuncs, self.config.POOLING)]]

    def __init__(self, config: Union[_YamlCFG, _JsonCFG]):  # remove CFG and refer to self
        """
        :param config:
        :type config:
        :param args:
        :type args:
        """
        super().__init__(config)
        if config.FILE_TYPE == "JSON":
            self.config = self._JsonCFG(config)
        elif config.FILE_TYPE == "YAML":
            self.config = self._YamlCFG(config)
        else:
            raise AttributeError("Config file type not supported")
        self.collect_funcs()
        use_bias = self.ACTIVATION_F == nn.InstanceNorm2d
        # TODO finish abstraction

        # assert len(self.cfg.LAYER_NUMS) == len(self.cfg.LAYER_STRIDES) == len(self.cfg.NUM_FILTERS)
        # layer_nums = self.cfg.LAYER_NUMS
        # layer_strides = self.cfg.LAYER_STRIDES
        # num_filters = self.cfg.NUM_FILTERS
        #
        # if self.cfg.UPSAMPLE_STRIDES is not None:
        #     assert len(self.cfg.UPSAMPLE_STRIDES) == len(self.cfg.NUM_UPSAMPLE_FILTERS)
        #     num_upsample_filters = self.cfg.NUM_UPSAMPLE_FILTERS
        #     upsample_strides = self.cfg.UPSAMPLE_STRIDES
        # else:
        #     upsample_strides = num_upsample_filters = []
        #
        # num_levels = len(layer_nums)
        # c_in_list = [self.cfg.INPUT_CHANNELS, *num_filters[:-1]]
        # self.blocks = nn.ModuleList()
        # self.deblocks = nn.ModuleList()
        #
        # # nn.ZeroPad2d(1)
        #
        # for lvl_i in range(num_levels):
        #     cur_layers = []
        #     for lr_ix in range(layer_nums[lvl_i]):
        #         self._conv_layer(cur_layers, lvl_i)
        #         self._norm_layer(cur_layers, lvl_i)
        #         self._activation_layer(cur_layers, lvl_i)
        #         self._pooling_layer(cur_layers, lvl_i, lr_ix)
        #
        #     self.blocks.append(nn.Sequential(*cur_layers))
        #     if len(upsample_strides) > 0:
        #         stride = upsample_strides[lvl_i]
        #         if stride >= 1:
        #             self.deblocks.append(nn.Sequential(
        #                 nn.ConvTranspose2d(
        #                     num_filters[lvl_i], num_upsample_filters[lvl_i],
        #                     upsample_strides[lvl_i],
        #                     stride=upsample_strides[lvl_i], bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[lvl_i], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #         else:
        #             stride = np.round(1 / stride).astype(np.int)
        #             self.deblocks.append(nn.Sequential(
        #                 nn.Conv2d(
        #                     num_filters[lvl_i], num_upsample_filters[lvl_i],
        #                     stride,
        #                     stride=stride, bias=False
        #                 ),
        #                 nn.BatchNorm2d(num_upsample_filters[lvl_i], eps=1e-3, momentum=0.01),
        #                 nn.ReLU()
        #             ))
        #
        # c_in = sum(num_upsample_filters)
        # if len(upsample_strides) > num_levels:
        #     self.deblocks.append(nn.Sequential(
        #         nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
        #         nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
        #         nn.ReLU(),
        #     ))
        #
        # self.num_bev_features = c_in

    def init_optimizer(self, optimizer: str = 'ADAM', epochs: int = 30):
        lr = self.lr
        if optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.parameters(), lr=lr, betas=(0.5, 0.9), weight_decay=0.0005)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9, dampening=0.9, weight_decay=0.001)
        eta_min = lr * (0.1 ** 3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=eta_min)

    def build(self):
        pass

    def _conv_layer(self, lvl: List[nn.Module], ix):
        lvl.extend(
            nn.Conv2d(self.config.NUM_FILTERS[ix], self.config.NUM_FILTERS[ix], kernel_size=3, padding=1, bias=False))

    def _norm_layer(self, lvl: List[nn.Module], ix):
        if self.NORMALIZATION_F is not None:
            lvl.extend(self.NORMALIZATION_F(self.config.NUM_FILTERS[ix], **self.config.NORM_ARGS[ix]))

    def _activation_layer(self, lvl: List[nn.Module], ix):
        if self.ACTIVATION_F is not None:
            lvl.extend(self.ACTIVATION_F(self.config.NUM_FILTERS[ix]))
        raise ValueError("Missing activation function specification 'ACTIVATION' in config file!")

    def _pooling_layer(self, lvl: List[nn.Module], blk_ix, layer_ix):
        if self.config.POOLING is not None:
            try:
                lvl.extend(self.POOLING_F(**self.config.POOL_ARGS[blk_ix][layer_ix]))
            except KeyError:
                return

    @staticmethod
    def get_config():
        return None

    def forward(self, *args):
        raise NotImplementedError("forward() method not implemented for Backbone 2D")

    def freeze_layers(self):
        for module, layers_ls in self.FREEZE_LAYERS:
            for i, (name, child) in enumerate(module.named_children()):
                if i in layers_ls:
                    for param in child.parameters():
                        param.requires_grad = False

            for name, param in module.named_parameters():
                print("> FROZEN LAYER:", name, not param.requires_grad)
