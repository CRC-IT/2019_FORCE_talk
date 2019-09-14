from sklearn.neighbors import KNeighborsRegressor
from .base_regressor import BaseRegressor


class KNeighborsRegressorModel(BaseRegressor):
    short_name = 'KNNR'
    sklearn_model = KNeighborsRegressor()
    hyper_param_dict = {'n_neighbors': [1, 2, 5, 10]}

    def __init__(self):
        super().__init__()
        self.set_params()

