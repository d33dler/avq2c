from datetime import datetime
from pathlib import Path
from typing import Tuple, Any, Sequence, List

import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import hp
from hyperopt.pyll import scope
from pandas import DataFrame
import matplotlib.pyplot as plt
from xgboost import DMatrix

from models.de_heads.dtree import DTree
from models.utilities.utils import load_config, DataHolder


class RandomForestHead(DTree):
    """
    XGBModel head wrapper class providing minimalistic interfacing with the model
    and automatic initialization using the local file config
    """

    output: np.ndarray = None
    config_id = 'config.yaml'

    def __init__(self, config):
        """
                Initialize model and create search space for parameter fine-tuning using hyperopt
                :param config:
                :type config:
                """
        super().__init__(config)
        t = self.config.TYPE

        self._init_model(t, **self.params)
        self.search_space = {
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 0.01),
            'max_depth': scope.int(hp.quniform('max_depth', 3, 11, 1)),
            'num_parallel_tree': scope.int(hp.quniform('num_parallel_tree', 30, 130, 5)),
            'num_boost_round': 1,
            'min_child_weight': hp.quniform('min_child_weight', 0.1, 5, 0.5),
            'subsample': hp.uniform('subsample', 0.5, 0.9),
            'gamma': hp.quniform('gamma', 0.1, 15, 1),
            'reg_lambda': hp.quniform('reg_lambda', 0.1, 10, 1),
            'objective': self.params['objective'],
            'eval_metric': self.params['eval_metric'],
            'tree_method': self.params['tree_method'],
            'eta': 1,
            'seed': 123,
        }

    @staticmethod
    def get_config():
        return load_config(Path(Path(__file__).parent / "config.yaml"))

    def plot_importance(self):
        xgb.plot_importance(self.model, importance_type='gain')
        plt.show(block=False)

    def plot_self(self):
        xgb.plot_tree(self.model)
        plt.show()
