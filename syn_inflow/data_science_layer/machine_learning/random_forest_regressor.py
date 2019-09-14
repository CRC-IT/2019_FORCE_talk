from sklearn.ensemble import RandomForestRegressor
from .base_regressor import BaseRegressor


class RandomForestRegressorModel(BaseRegressor):
    short_name = 'RFR'
    sklearn_model = RandomForestRegressor()
    hyper_param_dict = {'n_estimators': [1, 2, 5, 10, 50, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

