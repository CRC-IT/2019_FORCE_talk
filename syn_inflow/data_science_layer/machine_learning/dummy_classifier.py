from sklearn.dummy import DummyClassifier
from .base_classifier import BaseClassifier


class DummyClassifierModel(BaseClassifier):
    short_name = 'DC'
    sklearn_model = DummyClassifier()

    def __init__(self):
        super().__init__()
        self.set_params()
