from sklearn.linear_model import LogisticRegression
from .base_classifier import BaseClassifier


class LogisticRegressionClassifierModel(BaseClassifier):
    short_name = 'LRC'
    sklearn_model = LogisticRegression()
    hyper_param_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

