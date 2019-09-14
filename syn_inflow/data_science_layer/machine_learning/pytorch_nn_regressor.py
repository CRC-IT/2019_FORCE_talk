from .not_sk_learn_ml_models.pytorch_base import PytorchModel
from .base_regressor import BaseRegressor
import pkg_resources


class PytorchRegressor(BaseRegressor):
    sklearn_model = PytorchModel()
    short_name = 'Pytorch-Regressor - funnel NN'

    def __init__(self, model_class=None):
        super().__init__()
        if model_class is None:
            self.sklearn_model = PytorchModel()
        else:
            self.sklearn_model = model_class()
        self.set_default_model()

    def set_cached_model_path(self, model_name):
        path = pkg_resources.resource_filename(
            'crcdal', '/cache/'+model_name)
        self.sklearn_model.load_cached_model = path

    def set_model_save_path(self, model_name):
        path = pkg_resources.resource_filename(
            'crcdal', '/cache/' + model_name)
        self.sklearn_model.save_cached_model = path
