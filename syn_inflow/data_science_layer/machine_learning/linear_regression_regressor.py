from sklearn.linear_model import LinearRegression
from .base_regressor import BaseRegressor


class LinearRegressionRegressorModel(BaseRegressor):
    short_name = 'LR'
    sklearn_model = LinearRegression()

    def __init__(self):
        super().__init__()
        self.set_params()
        self.y_greater_than_one = False
    #
    # # def fit(self, x, y):
    # #     if y.shape[1] > 1:
    # #         self.y_greater_than_one = True
    # #         self.sklearn_model = [self.sklearn_model] * y.shape[1]
    # #         for y_idx in range(y.shape[1]):
    # #             self.sklearn_model[y_idx].fit(x, y[:, y_idx])
    # #     else:
    # #         super().fit(x, y)
    #
    # def predict(self, x) -> (object, object):
    #     output = []
    #     if self.y_greater_than_one:
    #         for y_idx in range(len(self.sklearn_model)):
    #             output.append(self.sklearn_model[y_idx].transform(x))
    #         return np.concatenate(output, axis=1)
    #     else:
    #         return super().predict(x)
    #
    # def search_models(self, x, y, **kwargs):
    #     if y.shape[1] > 1:
    #         self.y_greater_than_one = True
    #         self._fit(x, y)
    #         self.cv_search_models()
    #         # y_pred = self.predict(x)
    #         # self.best_model
    #     else:
    #         super().search_models(x, y)
    #
    # def _fit(self, x, y):
    #     if y.shape[1] > 1:
    #         if isinstance(y, pd.DataFrame):
    #             y = y.values
    #         self.y_greater_than_one = True
    #         self.sklearn_model = [self.sklearn_model] * y.shape[1]
    #         for y_idx in range(y.shape[1]):
    #             self.sklearn_model[y_idx].fit(x, y[:, y_idx])
    #     else:
    #         super()._fit(x, y)
