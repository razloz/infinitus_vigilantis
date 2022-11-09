"""Three blind mice to predict the future."""
import time
import torch
import traceback
import matplotlib.pyplot as plt
import source.ivy_commons as icy
from torch.nn import Linear, Module, MSELoss, ParameterDict, Transformer
from torch.nn.functional import gelu
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from math import sqrt, log, inf
from os import mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbol, *args, verbosity=1, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._features_ = ['open', 'high', 'low', 'close',
                           'vol_wma_price', 'price_wema']
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
        phi = 1.618033988749894
        self.iota = iota = 1 / 137
        self._lr_init_ = phi
        self._lr_max_ = iota / (phi - 1)
        self._lr_min_ = iota / phi
        self._n_dropout_ = ((137 * phi) ** iota) - 1
        self._n_features_ = len(self._features_)
        self._n_hidden_ = 64
        self._n_layers_ = 1024
        self._warm_steps_ = 512
        self.cauldron = torch.nn.GRU(
            input_size=self._n_features_,
            hidden_size=self._n_hidden_,
            num_layers=self._n_layers_,
            bias=True,
            batch_first=True,
            dropout=self._n_dropout_,
            bidirectional=False,
            )
        self.linear = Linear(
            in_features=self._n_hidden_,
            out_features=1,
            bias=True,
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
        self.loss_fn = MSELoss(reduction='mean')
        self._tolerance_ = 1e-4
        self._cook_time_ = 1800
        self.candles = None
        self.wax = None
        self.metrics = ParameterDict()
        self.predictions = ParameterDict()
        self.verbosity = int(verbosity)
        self.to(self._device_)
        self.quick_load()
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set features to', self._n_features_)
            print(self._prefix_, 'set hidden to', self._n_hidden_)
            print(self._prefix_, 'set layers to', self._n_layers_)
            print(self._prefix_, 'set dropout to', self._n_dropout_)
            print(self._prefix_, 'set initial lr to', self._lr_init_)
            print(self._prefix_, 'set max lr to', self._lr_max_)
            print(self._prefix_, 'set min lr to', self._lr_min_)
            print(self._prefix_, 'set warm steps to', self._warm_steps_)
            print(self._prefix_, 'set cook time to', self._cook_time_)
            print('')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            state_path = f'{self._state_path_}/{self._symbol_}'
            if call_type == 0:
                state = torch.load(
                    f'{state_path}.state',
                    map_location=self._device_type_,
                    )
                self.load_state_dict(state['moirai'])
                for key, value in state['predictions'].items():
                    self.predictions[key] = value
                for key, value in state['metrics'].items():
                    self.metrics[key] = value
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
                self.candles = torch.load(
                    f'{state_path}.candles',
                    map_location=self._device_type_,
                    )
                self.wax = torch.load(
                    f'{state_path}.wax',
                    map_location=self._device_type_,
                    )
            elif call_type == 1:
                torch.save(
                    {
                        'metrics': self.metrics,
                        'moirai': self.state_dict(),
                        'predictions': self.predictions,
                        },
                    f'{state_path}.state',
                    )
                torch.save(self.candles, f'{state_path}.candles')
                torch.save(self.wax, f'{state_path}.wax')
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

    def __time_plot__(self, predictions, adj_loss):
        epoch = self.metrics['epochs']
        x = range(predictions.shape[0])
        y_p = predictions.detach().cpu()
        y_t = self.wax.detach().cpu()
        n_p = round(y_p[-1].item(), 2)
        n_t = round(y_t[-1].item(), 2)
        plt.plot(x, y_p, label=f'Prediction: {n_p}', color='#FFE88E')
        plt.plot(x, y_t, label=f'Target: {n_t}', color='#FF9600')
        plt.suptitle(f'{self._symbol_} (epochs: {epoch})', fontsize=18)
        plt.legend(ncol=1, fancybox=True)
        ts = int(time.time())
        epochs_path = f'{self._epochs_path_}/{ts}.png'
        plt.savefig(epochs_path)
        plt.clf()
        if self.verbosity > 0:
            prefix = self._prefix_
            print(prefix, 'PREDICTIONS TAIL:', y_p[-1].item())
            print(prefix, 'TARGET TAIL:', y_t[-1].item())
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
            coated = 0
            coating_candles = True
            cook_time = self._cook_time_
            wax = self.wax
            batch_size = wax.shape[0]
            mean_wax = wax.mean().item()
            warm_steps = self._warm_steps_
            if 'bubbling_wax' in self.metrics.keys():
                heating_up = False
            else:
                heating_up = True
            tolerance = mean_wax * self._tolerance_
            n_heat = 0
            n_heat_avg = inf
            n_reheat = 137
            plot_state = True
            t_cook = time.time()
            self.metrics['epochs'] = 0
            self.metrics['loss'] = 0
            self.metrics['mae'] = 0
            self.metrics['mse'] = 0
            while coating_candles:
                coated += 1
                coated_candles = self.forward()
                loss = self.loss_fn(coated_candles, wax)
                loss.backward()
                adj_loss = (sqrt(loss.item()) / mean_wax)
                n_heat += adj_loss
                self.optimizer.step()
                if not heating_up:
                    self.schedule_cyclic.step()
                else:
                    if coated == warm_steps:
                        self.metrics['bubbling_wax'] = True
                        heating_up = False
                        self.schedule_cyclic.step()
                    else:
                        self.schedule_warm.step()
                delta = (coated_candles - wax).abs()
                correct = delta[delta >= tolerance]
                correct = delta[delta <= tolerance].shape[0]
                absolute_error = batch_size - correct
                self.metrics['acc'] = (correct, batch_size)
                self.metrics['epochs'] += 1
                self.metrics['loss'] += adj_loss
                self.metrics['mae'] += absolute_error
                self.metrics['mse'] += absolute_error ** 2
                if time.time() - t_cook >= cook_time:
                    if not heating_up:
                        coating_candles = False
                        epochs = self.metrics['epochs']
                        self.metrics['loss'] = self.metrics['loss'] / epochs
                        self.metrics['mae'] = self.metrics['mae'] / epochs
                        self.metrics['mse'] = sqrt(self.metrics['mse'])
                elif coated % n_reheat == 0:
                    n_heat = n_heat / n_reheat
                    if n_heat >= n_heat_avg:
                        heating_up = True
                        n_heat_avg = inf
                    else:
                        n_heat_avg = n_heat
                    n_heat = 0
                self.quick_save()
                if plot_state:
                    self.__time_plot__(coated_candles, adj_loss)
            if self.verbosity > 1: print('')
            return coated_candles.clone()
        else:
            self.eval()
            with torch.no_grad():
                coated_candles = self.forward()
            return coated_candles.clone()

    def collect(self, offering):
        """Takes the offered dataframe and converts it to candle tensors."""
        l_features = self._features_
        l_targets = self._target_feature_
        n_time = len(offering.index)
        params = self._p_tensor_
        tensor = torch.tensor
        if n_time % 2 != 0:
            n_time -= 1
        n_half = int(n_time / 2)
        candles = offering[l_features]
        candles = candles.to_numpy()[:n_half]
        candles = tensor(candles, **params)
        self.candles = candles.detach().cpu().requires_grad_(True)
        wax = offering[l_targets]
        wax = wax.to_list()[-n_half:]
        wax = tensor(wax, **params)
        self.wax = wax.detach().cpu()
        self.quick_save()
        return True

    def forward(self):
        """**bubble*bubble**bubble**"""
        candles = self.cauldron(self.candles)[0]
        candles = self.linear(candles)
        candles = candles.squeeze(1)
        candles = 1 - gelu(candles)
        print('candles shape:', candles.shape)
        print('candles head:', candles[:5])
        print('candles tail:', candles[-5:])
        print('candles mean:', self.candles.mean())
        candles = candles * self.candles.mean()
        return candles.clone()

    def predict(self, dataframe):
        """Take a batch of inputs and return the future signal."""
        return self.__time_step__(None, study=False)

    def quick_load(self):
        """Alias to load RNN state."""
        self.__manage_state__(call_type=0)

    def quick_save(self):
        """Alias to save RNN state."""
        self.__manage_state__(call_type=1)

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        return self.__time_step__(study=True)
