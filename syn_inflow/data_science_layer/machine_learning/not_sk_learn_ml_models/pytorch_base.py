import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import save
import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
import math
from collections import defaultdict
import pkg_resources


class PytorchModel(object):
    _gpu_available = torch.cuda.is_available()
    _gpus = torch.device("cuda")
    # _gpu_2 = torch.device("cuda:1")
    _cpu = torch.device("cpu")
    _batch_size = 100000
    _output_size = 1
    _model = None
    layer_count = 6
    hidden_layers = None
    default_hidden_layer_size = 800
    activation = nn.PReLU
    final_activation = nn.Sigmoid
    loss_func = torch.nn.SmoothL1Loss()
    allow_negative_predictions = True
    train_time = 3000
    training_curve = defaultdict(list)
    best_loss = 10000000000000000.0
    best_params_dict = {}
    use_cpu = False
    load_cached_model = None
    save_cached_model = 'model_v12'
    x_tensor = None
    y_tensor = None
    t_x = None
    x_y = None
    optimizer = None
    outputs = None
    batch_counter = 0
    counter = 0
    running_loss = 0.0
    fit_model = True

    def __init__(self):

        self.activation = nn.SELU
        self.layer_count = 20
        self.hidden_layers = [40] * self.layer_count
        self.train_layers = [True] * (self.layer_count * 2)
    # TODO Add in an update for layer count when the hidden layer size list is updated

    def fit(self, x, y, test_x=None, test_y=None):
        self.outputs, size = self._prep_input(x, y)
        if self.load_cached_model is not None:
            self.load_model(x, y, self.load_cached_model)
        if self.fit_model:
            self._fit(x, y)

    def load_model(self, x, y, path):
        device = self.get_device()
        self._model = torch.load(path)
        self._model.to(device)

    def _prep_tensors(self, x, y):
        x, y = self.handle_pandas(x, y)
        y = torch.tensor(y)
        x = torch.tensor(x, requires_grad=True)
        x = x.float()
        y = self.set_y_data_type(y)
        if self._gpu_available and not self.use_cpu:
            x.cuda().to(self._gpus)
            y.cuda().to(self._gpus)
        else:
            x.to(self._cpu)
            y.to(self._cpu)
        return x, y

    def _prep_input(self, x, y):
        size = y.shape[0]
        if y.ndim > 1:
            outputs = y.shape[1]
        else:
            outputs = 1
        if size < self._batch_size:
            self._batch_size = size
        if self._model is None:
            self._setup_model(x, y)
        return outputs, size

    def predict(self, x):
        if x.shape[0] > 50000:
            predictions = []
            split_size = int(x.shape[0] / 50000) + 1
            list_of_outputs = np.array_split(x, split_size)
            for output in list_of_outputs:
                predictions.append(self._predict(output))
            predictions = np.concatenate(predictions)
        else:
            predictions = self._predict(x)
        return predictions

    def _fit(self, x, y, test_x=None, test_y=None):
        for idx, param in enumerate(self._model.parameters()):
            param.requires_grad = self.train_layers[idx]
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._model.parameters()),
            lr=0.005, amsgrad=True)
        # scheduler = ReduceLROnPlateau(optimizer)

        self.x_tensor, self.y_tensor = self._prep_tensors(x, y)
        if test_x is not None:
            self.t_x, self.t_y = self._prep_tensors(test_x, test_y)
        permutation = torch.randperm(self.x_tensor.size()[0])
        last_loss = 100000000
        done = False
        path = self.get_path('start_test')
        torch.save(self._model, path)
        t = 0
        while True:
            t += 1
            last_loss = self.run_epoch(t, permutation, last_loss)
            if done:
                break
            if t > self.train_time:
                break
        self._model.load_state_dict(self.best_params_dict)
        # if self._gpu_available:
        #     self._model.to(self._gpus)
        print(((self.best_loss) ** (0.5)) / x.shape[0])
        path = self.get_path(self.save_cached_model)
        torch.save(self._model, path)
        self.load_cached_model = path

    def _predict(self, x):
        x, _ = self.handle_pandas(x)
        x = torch.tensor(x)
        x = x.float()
        if self._gpu_available:
            x.cuda().to(self._gpus)
            try:
                predictions = self._model(x.cuda())
                predictions = predictions.cpu()
            except Exception as ex:
                print(ex)
                self.load_model(None, None, self.load_cached_model)
                predictions = self._model(x)

        else:
            predictions = self._model(x)
        # predictions[predictions < 0] = 0
        return predictions.detach().numpy()

    def _setup_model(self, x, y):
        if y.ndim == 2:
            if y.shape[1] > 1:
                self.hidden_layers[len(self.hidden_layers)-1] = y.shape[1]
        modules = []
        previous_layer_size = x.shape[1]
        for x in range(self.layer_count):
            if self.hidden_layers is not None:
                layer_size = self.hidden_layers[x]
            else:
                layer_size = self.default_hidden_layer_size
            modules.append(nn.Linear(previous_layer_size, layer_size))
            if x == self.layer_count - 1:
                # modules.append(self.final_activation())
                pass
            else:
                # modules.append(nn.Dropout(p=0.001))
                modules.append(self.activation())
            previous_layer_size = layer_size
        self._model = nn.Sequential(*modules)
        if self._gpu_available:
            self._model = self._model.cuda().to(self._gpus)

    def run_epoch(self, t, permutation, last_loss):
        self.counter = 0
        self.running_loss = 0.0
        self.batch_counter = 0
        for i in range(0, self.x_tensor.size()[0], self._batch_size):
            self.run_training_iteration(i, permutation)
        if self.running_loss < self.best_loss:
            self.best_params_dict = self._model.state_dict()
            self.best_loss = self.running_loss
        print(t, self.running_loss)
        if self.running_loss - last_loss > -0.000001:
            if last_loss < 100000:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.97
        # elif abs((self.running_loss / self.counter) - last_loss) < 0.00000001:
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 1.0001
        # last_loss = self.running_loss / self.counter
        self.training_curve['iter'].append(t)
        self.training_curve['loss'].append(self.running_loss)
        last_loss = self.running_loss
        return last_loss

    def run_training_iteration(self, i, permutation):
        if i + self._batch_size > self.x_tensor.size()[0]:
            current_size = self.x_tensor.size()[0] - i
        else:
            current_size = self._batch_size
        self.batch_counter += self._batch_size

        indices = permutation[i:i + current_size]
        batch_x, batch_y = self.x_tensor[indices], self.y_tensor[indices, :]
        if self._gpu_available:
            predictions = self._model(batch_x.cuda())
        else:
            predictions = self._model(batch_x)
        if not self.allow_negative_predictions:
            predictions[predictions < 0] = 0
        if self._gpu_available:
            loss = self.loss_func(predictions,
                                  batch_y.cuda().view(current_size, self.outputs))
        else:
            loss = self.loss_func(predictions,
                                  batch_y.view(current_size,
                                                      self.outputs))
        if self.counter % 10 == 0:
            print(loss.item())
        self._model.zero_grad()
        self.optimizer.zero_grad()
        loss.backward()
        self.running_loss += loss.item()
        self.optimizer.step()

        self.counter += 1

    def get_params(self):
        return {}

    def _get_param_names(self):
        return {}

    def set_params(self, **kwargs):
        return self

    def set_y_data_type(self, y):
        return y.float()

    def get_device(self):
        if self.use_cpu:
            device = self._cpu
        elif self._gpu_available:
            device = self._gpus
        else:
            device = self._cpu
        return device

    def handle_pandas(self, x, y=None):
        if not isinstance(x, pd.DataFrame) and not isinstance(x, pd.Series):
            if y is not None:
                if len(y.shape) == 1:
                    if isinstance(y, pd.Series):
                        y = y.values
                    y = y.reshape([y.shape[0], 1])
                else:
                    y = y.values
            return x, y
        if isinstance(x, pd.DataFrame):
            x = x.values
            if y is not None:
                y = y.values
        return x, y

    def get_path(self, modifier):
        # path = pkg_resources.resource_filename('crcdal', '/cache/'+modifier)
        return modifier

