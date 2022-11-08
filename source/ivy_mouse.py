"""Three blind mice to predict the future."""
import time
import torch
import traceback
import matplotlib.pyplot as plt
import source.ivy_commons as icy
from torch.nn import Module, MSELoss, ParameterDict, Transformer
from torch.nn.functional import gelu
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from math import sqrt, log, inf
from os import mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
# global constants
mu = 0.000001
alpha = 0.0072973525693
gamma = 0.577215664901532
whisper = 1.054571817
phi = 1.618033988749894
epsilon = 2.71828
pi = 3.141592653589793
tau = 137
zeta = 1 / tau
xi_lower = (gamma ** phi) * (whisper ** phi) * (alpha ** phi)
xi_upper = 1 - (mu ** (sqrt(xi_lower / (tau ** phi)) * (pi ** phi)))
euler_golden_pi = (-log(phi) * -log(pi)) ** epsilon


class ThreeBlindMice(Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbol, batch_size=34, verbosity=1, *args, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._features_ = [
            'open', 'high', 'low', 'close', 'vol_wma_price',
            'fib_retrace_0.236', 'fib_retrace_0.382', 'fib_retrace_0.5',
            'fib_retrace_0.618', 'fib_retrace_0.786', 'fib_retrace_0.886',
            'price_wema', 'price_dh', 'price_dl', 'price_mid',
            ]
        self._target_feature_ = 'price_med'
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._symbol_ = str(symbol).upper()
        self._batch_size_ = int(batch_size)
        self._lr_init_ = xi_upper
        self._lr_max_ = alpha
        self._lr_min_ = xi_lower
        self._n_dropout_ = whisper - 1
        self._n_features_ = len(self._features_)
        self._n_hidden_ = (16 ** 3) * 2
        self._n_layers_ = tau
        self._n_norm_eps_ = tau * mu * phi
        self._warm_steps_ = int(self._n_layers_ * 3)
        self.cauldron = Transformer(
            d_model=self._n_features_,
            nhead=3,
            num_encoder_layers=self._n_layers_,
            num_decoder_layers=self._n_layers_,
            dim_feedforward=self._n_hidden_,
            dropout=self._n_dropout_,
            activation='gelu',
            layer_norm_eps=self._n_norm_eps_,
            batch_first=True,
            norm_first=True,
            **self._p_tensor_,
            )
        self.optimizer = RMSprop(
            self.cauldron.parameters(),
            lr=self._lr_init_,
            foreach=True,
            )
        self.schedule_warm = CosineAnnealingWarmRestarts(
            self.optimizer,
            self._warm_steps_,
            eta_min=self._lr_min_,
            )
        self.schedule_cyclic = CyclicLR(
            self.optimizer,
            self._lr_min_,
            self._lr_max_,
            )
        self.loss_fn = MSELoss(reduction='sum')
        self._tolerance_ = alpha
        self.inputs = None
        self.targets = None
        self.cauldron_targets = None
        self.tensors = dict(coated=None, sealed=[], inputs=None, targets=None)
        self.metrics = ParameterDict()
        self.predictions = ParameterDict()
        self.verbosity = int(verbosity)
        self.wax = zeta
        self.to(self._device_)
        self.quick_load()
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set batch_size to', self._batch_size_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set features to', self._n_features_)
            print(self._prefix_, 'set hidden dim to', self._n_hidden_)
            print(self._prefix_, 'set layers to', self._n_layers_)
            print(self._prefix_, 'set dropout to', self._n_dropout_)
            print(self._prefix_, 'set norm eps to', self._n_norm_eps_)
            print(self._prefix_, 'set initial learning rate to', self._lr_init_)
            print(self._prefix_, 'set max learning rate to', self._lr_max_)
            print(self._prefix_, 'set min learning rate to', self._lr_min_)
            print(self._prefix_, 'set warm steps to', self._warm_steps_)
            print('')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            state_path = f'{self._state_path_}/{self._symbol_}.state'
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

    def __time_plot__(self, outputs, targets, adj_loss):
        epoch = self.metrics['epochs']
        x = range(self._batch_size_)
        y_p = outputs.detach().cpu()
        y_t = targets.detach().cpu()
        plt.plot(x, y_p, label=f'Prediction: {y_p[-1].item()}', color='#FFE88E')
        plt.plot(x, y_t, label=f'Target: {y_t[-1].item()}', color='#FF9600')
        plt.suptitle(f'{self._symbol_} (epochs: {epoch})', fontsize=18)
        plt.legend(ncol=1, fancybox=True)
        ts = int(time.time())
        epochs_path = f'{self._epochs_path_}/{ts}.png'
        plt.savefig(epochs_path)
        plt.clf()
        if self.verbosity > 0:
            prefix = self._prefix_
            print(prefix, 'OUTPUT TAIL:', outputs[-1].item())
            print(prefix, 'TARGET TAIL:', targets[-1].item())
            print(prefix, 'ADJ_LOSS:', adj_loss)
            print(prefix, 'ACC:', self.metrics['acc'])
            for k in ['loss', 'mae', 'mse']:
                v = self.metrics[k]
                if k == 'mse':
                    print(prefix, f'{k.upper()}: {sqrt(v)}')
                else:
                    print(prefix, f'{k.upper()}: {v / epoch}')
            last_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            print(prefix, 'LR:', '{:.10f}'.format(last_lr))
            print(prefix, 'EPOCH:', epoch, '\n')

    def __time_step__(self, study=False, plot=True):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        if study:
            self.train()
            self.optimizer.zero_grad()
            batch_size = self._batch_size_
            coated = 0
            coating_candles = True
            cook_time = self._n_layers_
            last_avg = inf
            running_sum = 0
            targets = self.targets
            mean_targets = targets.mean().item()
            warm_steps = self._warm_steps_
            heating_up = True
            tolerance = mean_targets * self._tolerance_
            n_plot = 50
            cauldron_targets = self.cauldron_targets
            while coating_candles:
                coated += 1
                coated_candles = self.forward()
                loss = self.loss_fn(coated_candles, cauldron_targets)
                loss.backward()
                adj_loss = (sqrt(loss.item()) / mean_targets) / batch_size
                self.optimizer.step()
                if not heating_up:
                    self.schedule_cyclic.step()
                else:
                    if coated == warm_steps:
                        heating_up = False
                        self.schedule_cyclic.step()
                    else:
                        self.schedule_warm.step()
                running_sum += adj_loss
                predictions = coated_candles[:, :4].mean(1)
                delta = (predictions - targets).abs()
                correct = delta[delta >= tolerance]
                correct = delta[delta <= tolerance].shape[0]
                absolute_error = batch_size - correct
                self.metrics['acc'] = (correct, batch_size)
                self.metrics['epochs'] += 1
                self.metrics['loss'] += adj_loss
                self.metrics['mae'] += absolute_error
                self.metrics['mse'] += absolute_error ** 2
                if coated % n_plot == 0:
                    self.quick_save()
                    self.__time_plot__(predictions, targets, adj_loss)
                if coated % cook_time == 0:
                    running_sum = running_sum / cook_time
                    if running_sum < last_avg:
                        last_avg = running_sum
                    else:
                        coating_candles = False
                        self.quick_save()
                        self.__time_plot__(predictions, targets, adj_loss)
            if self.verbosity > 1: print('')
            return predictions.clone()
        else:
            self.eval()
            with torch.no_grad():
                predictions = self.forward()[:, :4].mean(1)
            return predictions.clone()

    def forward(self):
        """**bubble*bubble**bubble**"""
        candles = self.cauldron(self.inputs, self.cauldron_targets)
        return gelu(candles).clone()

    def predict(self, dataframe):
        """Take a batch of inputs and return the future signal."""
        features = dataframe[self._features_][:batch_size].to_numpy()
        self.inputs = torch.tensor(features, **self._p_tensor_)
        self.inputs.requires_grad_(True)
        return self.__time_step__(None, study=False)

    def quick_load(self):
        """Alias to load RNN state."""
        self.__manage_state__(call_type=0)

    def quick_save(self):
        """Alias to save RNN state."""
        self.__manage_state__(call_type=1)

    def research(self, dataframe):
        """Moirai research session, fully stocked with cheese and drinks."""
        batch_size = self._batch_size_
        features = dataframe[self._features_][:batch_size].to_numpy()
        self.inputs = torch.tensor(features, **self._p_tensor_)
        self.inputs.requires_grad_(True)
        c_t = dataframe[self._features_][-batch_size:].to_numpy()
        self.cauldron_targets = torch.tensor(c_t, **self._p_tensor_)
        targets = dataframe[self._target_feature_][-batch_size:].to_numpy()
        self.targets = torch.tensor(targets, **self._p_tensor_)
        return self.__time_step__(study=True)

