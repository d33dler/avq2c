import time
from abc import ABC
from typing import Sequence, Tuple, Any, List

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from easydict import EasyDict
from hyperopt import STATUS_OK, fmin, tpe
from pandas import DataFrame
from scipy.special import softmax
from skimage.color import rgb2hsv
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix

from models.de_heads.dengine import DecisionEngine


class DTree(DecisionEngine, ABC):
    """
    Decision Tree decision engine type
    """
    all_features: List[str] = []
    search_space = {}
    _optim_cycles = 100
    _optim_threshold = 0.1

    _float_int_fix = ['max_depth', 'num_parallel_tree', 'num_boost_round']

    def __init__(self, config: EasyDict):
        super().__init__(config)
        self.fine_tuning = self.config.OPTIMIZE
        self.params: dict = self.config.PARAMETERS
        self._optim_cycles = self.config.OPTIMIZATION_ROUNDS
        self.optimizer = None
        self.ranks = None
        self.num_classes = None
        self._scaler = StandardScaler()
        self._cols = []
        self.bins = 8
        if config.TYPE == "CLASSIFIER":
            self.num_classes = config.PARAMETERS["num_class"]
            self.ranks = self.num_classes

    def optimize(self, train_X: pd.DataFrame, train_Y: pd.DataFrame, eval_set: Tuple[Any, Any]):
        """
        Hyperparameter finetuning using hyperopt for ensemble tree (gradient boosting) algorithms
        Uses mlflow for logging & tracking progress & performance.
        :param eval_set:
        :param train_X:
        :type train_X:
        :param train_Y:
        :type train_Y:
        :return:
        :rtype:
        """
        X_train, _, y_train, _ = train_test_split(train_X, train_Y, test_size=0.3, shuffle=True)
        X_test, y_test = eval_set
        y_test = y_test.astype(int)
        y_train = y_train.astype(int)
        train = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=True)
        test = xgb.DMatrix(data=X_test, label=y_test, enable_categorical=True)
        labels = np.unique(y_test)
        mlflow.xgboost.autolog(disable=True, silent=True)

        def train_model(params):
            # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.

            # However, we can log additional information by using an MLFlow tracking context manager

            # Train model and record run time
            start_time = time.time()
            booster = xgb.train(params=params, dtrain=train, evals=[(test, "test")], verbose_eval=False)
            run_time = time.time() - start_time

            # Record Log loss as primary loss for Hyperopt to minimize
            predictions_test = booster.predict(test, output_margin=True)
            loss = log_loss(y_test, predictions_test, labels=labels)

            return {'status': STATUS_OK, 'loss': loss, 'booster': booster.attributes()}
        p = self.search_space.copy()
        p.pop('num_boost_round')
        with mlflow.start_run(run_name='xgb_loss_threshold'):
            best_params = fmin(
                fn=train_model,
                space=p,
                algo=tpe.suggest,
                loss_threshold=self._optim_threshold,
                max_evals=self._optim_cycles,
                rstate=np.random.default_rng(666),
            )
        mlflow.xgboost.autolog(disable=True)
        mlflow.end_run()
        [best_params.update({k: int(v)}) for k, v in best_params.items() if k in self._float_int_fix]
        return best_params

    def _create_input(self, matrix: np.ndarray):
        return pd.DataFrame(matrix, columns=self.features[self.BASE_FT])

    def load(self, state: dict):
        if isinstance(state, bytearray):
            return
        self._init_model(state['type'], **self.params)
        self.model.load_model(state['model'])
        self.params.update(state['hp'])
        self.features = state['features']
        self.is_fit = True

    def dump(self):
        return {
            "model": self.model.save_raw(),
            "hp": self.params,
            "type": self.config.TYPE,
            "features": self.features
        }

    def _init_model(self, _type: str, **params):
        self.model = xgb.Booster()
        if _type == 'REGRESSOR':
            pass
        elif _type == 'CLASSIFIER':
            self.search_space['num_class'] = self.params['num_class'] = self.num_classes
        else:
            raise NotImplementedError('XGB specified model type not supported')
        print(self.params)

    def forward(self, data: DataFrame):
        """
        Predict
        :return:
        :rtype:
        """
        output = self.model.predict(data=DMatrix(data[self._cols], enable_categorical=True))
        return output

    @staticmethod
    def normalize(x: np.ndarray, axis=1):
        """Softmax function with x as input vector."""
        return softmax(x, axis=axis)

    def _fit_model(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        self._cols = x.columns
        if eval_set is None:
            _, eval_x, _, eval_y = train_test_split(x, y, test_size=0.3, shuffle=True)
            eval_set = [(xgb.DMatrix(data=eval_x, label=eval_y, enable_categorical=True), 'test')]
        else:
            eval_set = [(xgb.DMatrix(data=eval_x, label=eval_y, enable_categorical=True), 'test') for eval_x, eval_y in
                        eval_set]
        self.model = xgb.train(params=self.params, dtrain=DMatrix(data=x, label=y, enable_categorical=True),
                               evals=eval_set,
                               num_boost_round=self.params.get('num_boost_round', 100), early_stopping_rounds=10,
                               **kwargs)

        [print(f">{o[0]} : {o[1]}") for o in
         sorted(self.model.get_score(importance_type='gain').items(), key=lambda q: q[1], reverse=True)]
        return self.model

    def fit(self, x: DataFrame, y: DataFrame, eval_set: Sequence[Tuple[Any, Any]], **kwargs):
        self.is_fit = True
        if self.fine_tuning:
            self.params.update(self.optimize(x, y, eval_set[0]))
            print(">BEST PARAMETERS:", self.params)
        self.features[self.ALL_FT] = [f for f in x.columns if f not in y.columns]
        return self._fit_model(x, y, eval_set, **kwargs)

    def feature_engineering(self, matrix: np.ndarray, _input, **kwargs):

        tree_df: pd.DataFrame = self._create_input(matrix)

        base_ft = self.features[self.BASE_FT]
        rank_ft = self.features[self.RANK_FT]
        misc_ft = self.features[self.MISC_FT]
        cls_vals = tree_df[base_ft].to_numpy()
        rank_indices = np.argsort(-cls_vals, axis=1)
        tree_df[rank_ft] = rank_indices

        tree_df['max'] = tree_df[base_ft].max(axis=1)
        tree_df['mean'] = tree_df[base_ft].mean(axis=1)
        tree_df['std'] = tree_df[base_ft].std(axis=1)

        tree_df[rank_ft] = tree_df[rank_ft].astype('int')
        # assume image_tensor is a torch tensor of shape [50, 3, 100, 100]
        batch_size, num_channels, height, width = _input.shape
        # reshape the tensor to [50, 3, 10000] to simplify computation
        image_tensor_reshaped = _input.view(batch_size, num_channels, height * width)
        # convert the tensor to numpy array
        image_array = image_tensor_reshaped.numpy()
        # convert from channel-first to channel-last format
        image_array = np.transpose(image_array, (0, 2, 1))
        # convert the images from RGB to HSV color space
        image_array = rgb2hsv(image_array)
        # extract color histogram features for each image
        histograms = []
        bins = self.bins
        for i in range(batch_size):
            hist_r, _ = np.histogram(image_array[i, :, 0], bins=bins, range=(0, 1))
            hist_g, _ = np.histogram(image_array[i, :, 1], bins=bins, range=(0, 1))
            hist_b, _ = np.histogram(image_array[i, :, 2], bins=bins, range=(0, 1))
            histograms.append(np.concatenate([hist_r, hist_g, hist_b]))

        # scale the features
        histograms = np.asarray(self._scaler.fit_transform(histograms))
        tree_df = pd.concat([tree_df, pd.DataFrame(histograms, columns=misc_ft)], axis=1)
        return tree_df
