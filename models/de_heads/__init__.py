from .xgboost.xgb import XGBHead
from .catboost.catb import CatBoostHead
from .lightgbm.lgbm import LightGBMHead
from .random_forest.rforest import RandomForestHead

__all__ = {
    'CatBoost': CatBoostHead,
    'XGB': XGBHead,
    'LightGBM': LightGBMHead,
    'RandomForest': RandomForestHead
}
