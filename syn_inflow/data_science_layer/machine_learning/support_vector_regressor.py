from sklearn.svm import SVR
from .base_regressor import BaseRegressor


class SupportVectorRegressorModel(BaseRegressor):
    short_name = 'SVR'
    sklearn_model = SVR()
    hyper_param_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

