"""Three blind mice to predict the future."""
import time
import torch
import traceback
import matplotlib.pyplot as plt
import source.ivy_commons as icy
from torch.nn import GRU, Linear, Module, MSELoss, ParameterDict
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as WarmRestarts
from math import sqrt, log, inf
from os import mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
# global constants
µ = 0.000001
α = 0.0072973525693
γ = 0.577215664901532
ħ = 1.054571817
φ = 1.618033988749894
ε = 2.71828
π = 3.141592653589793
τ = 137
εφπ = (-log(φ) * -log(π)) ** ε


class ThreeBlindMice(Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, batch_size=34, verbosity=0, *args, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._features_ = [
            'open', 'high', 'low', 'close', 'vol_wma_price',
            'fib_retrace_0.236', 'fib_retrace_0.382', 'fib_retrace_0.5',
            'fib_retrace_0.618', 'fib_retrace_0.786', 'fib_retrace_0.886',
            'price_wema', 'price_dh', 'price_dl', 'price_mid',
            ]
        self._targets_ = 'price_med'
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self.verbosity = int(verbosity)
        self._batch_size_ = int(batch_size)
        self._n_features_ = len(self._features_)
        self._n_hidden_ = self._n_features_ ** 2
        self._n_layers_ = self._batch_size_
        self._tolerance_ = α
        self._dropout_ = εφπ
        self._lr_ = γ
        self.inputs = None
        self.torch_gate = GRU(
            input_size=self._n_features_,
            hidden_size=self._n_hidden_,
            num_layers=self._n_layers_,
            bias=True,
            batch_first=True,
            dropout=self._dropout_,
            device=self._device_,
            )
        self.torch_loss = MSELoss()
        self.torch_linear = Linear(self._n_hidden_, 1, **self._p_tensor_)
        self.optimizer = Adam(self.parameters(), lr=self._lr_, foreach=True)
        self.scheduler = WarmRestarts(self.optimizer, T_0=τ, eta_min=µ)
        self.tensors = dict(coated=None, sealed=[], inputs=None, targets=None)
        self.metrics = ParameterDict()
        self.predictions = ParameterDict()
        self.wax = τ
        self.to(self._device_)
        self.__manage_state__(call_type=0)
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set n_features to', self._n_features_)
            print(self._prefix_, 'set batch_size to', self._batch_size_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set dropout to', self._dropout_)
            print(self._prefix_, 'set learning rate to', self._lr_)
            print(self._prefix_, 'set n_hidden to', self._n_hidden_)
            print(self._prefix_, 'set n_layers to', self._n_layers_)
            print('')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            state_path = f'{self._state_path_}/moirai.state'
            if call_type == 0:
                state = torch.load(state_path, map_location=self._device_type_)
                self.load_state_dict(state['moirai'])
                for key, value in state['predictions'].items():
                    self.predictions[key] = value
                for key, value in state['metrics'].items():
                    self.metrics[key] = value
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            elif call_type == 1:
                torch.save(
                    {
                        'metrics': self.metrics,
                        'moirai': self.state_dict(),
                        'predictions': self.predictions,
                        },
                    state_path,
                    )
                if self.verbosity > 2:
                    print(self._prefix_, 'Saved RNN state.')
        except FileNotFoundError:
            self.metrics['acc'] = (0, 0)
            for key in ['epochs', 'loss', 'mae', 'mse']:
                self.metrics[key] = 0
        except Exception as details:
            if self.verbosity > 0:
                print(self._prefix_, 'Encountered an exception.')
                traceback.print_exception(details)

    def __time_step__(self, targets, study=False, plot=True):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        if study:
            self.train()
            self.optimizer.zero_grad()
            batch_size = self._batch_size_
            n_features = self._n_features_
            best_grad = None
            coated = 0
            coating_candles = True
            last_seal = inf
            least_loss = inf
            loss_size = int((batch_size * n_features) / 2)
            loss_stop = loss_size - 1
            losses = [inf for _ in range(loss_size)]
            confirmed = 0
            print('targets tail:', targets[-3:])
            print('starting loss loop:', loss_size)
            while True:
                outputs = self.forward()
                #print('outputs head:', outputs[:3])
                #print('outputs tail:', outputs[-3:])
                loss = self.torch_loss(outputs, targets)
                loss.backward()
                print('loss:', loss.item())
                self.optimizer.step()
                self.scheduler.step()
                if coating_candles == False:
                    break
                n_loss = loss.item()
                losses[coated] = n_loss
                if n_loss < least_loss:
                    best_grad = self.inputs.grad.detach()
                    print('best_grad head:', best_grad[:3])
                    print('best_grad tail:', best_grad[-3:])
                    print('least_loss:', least_loss)
                    least_loss = n_loss
                if coated == loss_stop:
                    coated = 0
                    sealed = sum(losses) / loss_size
                    if sealed < last_seal:
                        confirmed = 0
                        last_seal = sealed
                    else:
                        confirmed += 1
                        if confirmed == 3 and n_loss < 0:
                            coating_candles = False
                            self.inputs.grad = best_grad
                else:
                    coated += 1
            tolerance = self.inputs.mean() * self._tolerance_
            delta = (outputs - targets).abs()
            correct = delta[delta >= tolerance]
            correct = delta[delta <= tolerance].shape[0]
            absolute_error = batch_size - correct
            self.metrics['acc'] = (correct, batch_size)
            self.metrics['epochs'] += 1
            self.metrics['loss'] += loss.item()
            self.metrics['mae'] += absolute_error
            self.metrics['mse'] += absolute_error ** 2
            predictions = self.wax * outputs
            if plot:
                epoch = self.metrics['epochs']
                x = range(batch_size)
                y = predictions.detach().cpu()
                y_lbl = f'Prediction: {y[-1].item()}'
                print(y_lbl)
                plt.plot(x, y, label=y_lbl, color='#FFE88E')
                y = (self.wax * targets).detach().cpu()
                y_lbl = f'Target: {y[-1].item()}'
                print(y_lbl)
                plt.plot(x, y, label=y_lbl, color='#FF9600')
                plt.suptitle(f'Epoch: {epoch}', fontsize=18)
                plt.legend(ncol=1, fancybox=True)
                ts = int(time.time())
                epochs_path = f'{self._epochs_path_}/{ts}.png'
                plt.savefig(epochs_path)
                plt.clf()
            if self.verbosity > 1:
                prefix = self._prefix_
                print(prefix, 'ACC:', self.metrics['acc'])
                print(prefix, 'EPOCH:', epoch)
                for key in ['loss', 'mae', 'mse']:
                    value = self.metrics[key]
                    if key == 'mse':
                        print(prefix, f'{key.upper()}: {sqrt(value)}')
                    else:
                        print(prefix, f'{key.upper()}: {value / epoch}')
                print('')
            return predictions.clone()
        else:
            self.eval()
            with torch.no_grad():
                predictions = self.wax * ((-1 * self.forward()) ** 2)
            return predictions.clone()

    def forward(self):
        """**bubble*bubble**bubble**"""
        predictions = self.torch_gate(self.inputs)[1]
        predictions = self.torch_linear(predictions)
        return predictions.clone()

    def predict(self, dataframe):
        """Take a batch of inputs and return the future signal."""
        features = dataframe[self._features_].to_numpy()
        self.inputs = torch.tensor(features, **self._p_tensor_)
        self.inputs = (self.inputs / self.wax).requires_grad_(True)
        return self.__time_step__(None, study=False)

    def research(self, dataframe):
        """Moirai research session, fully stocked with cheese and drinks."""
        features = dataframe[self._features_].to_numpy()
        self.inputs = torch.tensor(features, **self._p_tensor_)
        self.inputs = (self.inputs / self.wax).requires_grad_(True)
        targets = dataframe[self._targets_].to_numpy()
        targets = torch.tensor(targets, **self._p_tensor_)
        targets = targets.expand(1, targets.shape[0]).H
        targets = targets / self.wax
        print('target shape:', targets.shape)
        self.__manage_state__(call_type=1)
        return self.__time_step__(targets, study=True)

