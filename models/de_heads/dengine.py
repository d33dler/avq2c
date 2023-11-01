from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Any, Dict, List

import numpy as np
from pandas import DataFrame

from models.interfaces.arch_module import ARCH


class DecisionEngine(ARCH.Child):
    """
    Decision engine superclass interface
    """
    ALL_FT = "*"
    BASE_FT = "base"
    MISC_FT = "misc"
    RANK_FT = "rank"
    features: Dict[str, List[str]] = dict({ALL_FT: [], BASE_FT: [], MISC_FT: [], RANK_FT: []})

    def __init__(self, config):
        super(DecisionEngine, self).__init__(config)
        self._enabled = False
        self._is_fit = False
        self.model: Any = None

    @abstractmethod
    def fit(self, x: DataFrame, y: DataFrame, eval_set: Tuple[Any, Any], **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, data: DataFrame):
        raise NotImplementedError

    @abstractmethod
    def plot_self(self):
        pass

    @abstractmethod
    def optimize(self, train_X, train_Y, eval_set):
        raise NotImplementedError

    @property
    def is_fit(self):
        return self._is_fit

    @is_fit.setter
    def is_fit(self, val: bool):
        self._is_fit = val

    @property
    def enabled(self):
        return self._enabled and self._is_fit

    def _create_input(self, matrix: np.ndarray):
        raise NotImplementedError

    def feature_engineering(self, matrix: np.ndarray, input, **kwargs):
        raise NotImplementedError

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = val
