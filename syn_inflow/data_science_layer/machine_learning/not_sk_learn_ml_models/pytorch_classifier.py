import torch
import torch.nn as nn
from .pytorch_base import PytorchModel

class PytorchCl(PytorchModel):
    final_activation = nn.LogSoftmax
    loss_func = torch.nn.CrossEntropyLoss()

    def __init__(self):
        super().__init__()
        self.final_activation = nn.LogSoftmax

    def predict(self, x):
        x = torch.tensor(x, requires_grad=True)
        x = x.float()
        predictions = self._model(x)
        _, predicted_class = torch.max(predictions, 1)
        return predicted_class.numpy()

    def predict_proba(self, x):
        x = torch.tensor(x, requires_grad=True)
        x = x.float()
        predictions = self._model(x)
        _, predicted_class = torch.max(predictions, 1)
        return predictions.detach().numpy()

    def set_y_data_type(self, y):
        return y.long()

