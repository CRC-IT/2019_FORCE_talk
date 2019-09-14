from sklearn.neural_network import MLPClassifier
from .base_classifier import BaseClassifier


class MultiLayerPerceptronClassifierModel(BaseClassifier):
    short_name = 'MLPC'
    sklearn_model = MLPClassifier()
    hyper_param_dict = {'hidden_layer_sizes': [[100,100], [300,300], [500,500]]}

    def __init__(self):
        super().__init__()
        self.set_params()

