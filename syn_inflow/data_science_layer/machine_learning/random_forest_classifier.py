from sklearn.ensemble import RandomForestClassifier
from .base_classifier import BaseClassifier


class RandomForestClassifierModel(BaseClassifier):
    short_name = 'RFC'
    sklearn_model = RandomForestClassifier()
    hyper_param_dict = {'n_estimators': [1, 2, 5, 10, 50, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

