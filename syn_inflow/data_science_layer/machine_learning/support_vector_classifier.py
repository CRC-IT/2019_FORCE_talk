from sklearn.svm import SVC
from .base_classifier import BaseClassifier


class SupportVectorClassifierModel(BaseClassifier):
    short_name = 'SVC'
    sklearn_model = SVC()
    hyper_param_dict = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

    def __init__(self):
        super().__init__()
        self.set_params()

