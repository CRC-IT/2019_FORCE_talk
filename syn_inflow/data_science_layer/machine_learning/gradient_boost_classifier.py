from sklearn.ensemble import GradientBoostingClassifier
from .base_classifier import BaseClassifier


class GradientBoostClassifierModel(BaseClassifier):
    short_name = 'GBC'
    sklearn_model = GradientBoostingClassifier()
    hyper_param_dict = {'n_estimators': [1, 2, 5, 10, 50, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

