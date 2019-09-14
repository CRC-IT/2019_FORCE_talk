'''
Authored by Nathaniel Jones, 
Modified and maintained by the Big Data Analytics Team
Copyright California Resources Corporation 2018, all rights reserved
'''

from crcdal.data_science_layer.machine_learning.not_sk_learn_ml_models.pytorch_classifier import PytorchCl
from .base_classifier import BaseClassifier


class PytorchMl(BaseClassifier):
    sklearn_model = PytorchCl()
    short_name = 'Pytorch-Classifier - funnel NN'

    def __init__(self, model_class=None):
        super().__init__()
        if model_class is None:
            self.sklearn_model = PytorchCl()
        else:
            self.sklearn_model = model_class()
        self.set_default_model()
