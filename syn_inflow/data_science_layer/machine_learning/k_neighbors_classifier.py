from sklearn.neighbors import KNeighborsClassifier
from .base_classifier import BaseClassifier


class KNeighborsClassifierModel(BaseClassifier):
    short_name = 'KNNC'
    sklearn_model = KNeighborsClassifier()
    hyper_param_dict = {'n_neighbors': [1, 2, 5, 10]}

    def __init__(self):
        super().__init__()
        self.set_params()

