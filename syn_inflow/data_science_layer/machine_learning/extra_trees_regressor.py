from sklearn.ensemble import ExtraTreesRegressor
from .base_regressor import BaseRegressor


class ExtraTreesRegressorModel(BaseRegressor):
    short_name = 'ETR'
    sklearn_model = ExtraTreesRegressor()
    hyper_param_dict = {'n_estimators': [1, 2, 5, 10, 50, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()




