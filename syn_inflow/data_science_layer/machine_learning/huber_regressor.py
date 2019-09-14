
from sklearn.linear_model import HuberRegressor
from .linear_regression_regressor import LinearRegressionRegressorModel


class HuberRegressorModel(LinearRegressionRegressorModel):
    short_name = 'Huber'
    sklearn_model = HuberRegressor()

    def __init__(self):
        super().__init__()
        self.set_params()