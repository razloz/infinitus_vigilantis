"""Three blind mice to predict the future."""
import time
import torch
import traceback
import matplotlib.pyplot as plt
import source.ivy_commons as icy
from torch import nn
from torch.nn.functional import gelu
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from math import sqrt, log, inf
from os import mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbol, *args, verbosity=1, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._batch_size_ = 5
        self._candles_ = ['open', 'high', 'low', 'close']
        self._cook_time_ = 1800
        self._c_iota_ = iota = 1 / 137
        self._c_phi_ = phi = 1.618033988749894
        self._delta_ = 'delta'
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._lr_init_ = phi
        self._lr_max_ = iota / (phi - 1)
        self._lr_min_ = iota / phi
        self._n_dropout_ = ((137 * phi) ** iota) - 1
        self._n_inputs_ = len(self._candles_)
        self._n_hidden_ = 3
        self._n_kernel_ = 9
        self._n_layers_ = 3 ** 7
        self._n_stride_ = 6
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = str(symbol).upper()
        self._targets_ = 'price_med'
        self._tolerance_ = 1e-4
        self._warm_steps_ = 256
        self._wax_ = 'price_wema'
        self.cauldron = nn.GRU(
            input_size=self._n_inputs_,
            hidden_size=self._n_hidden_,
            num_layers=self._n_layers_,
            bias=True,
            batch_first=True,
            dropout=self._n_dropout_,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='sum',
            delta=1.0,
            )
        self.normalizer = nn.InstanceNorm1d(
            self._n_inputs_,
            eps=1e-5,
            momentum=0.1,
            **self._p_tensor_,
            )
        self.optimizer = RMSprop(
            self.cauldron.parameters(),
            lr=self._lr_init_,
            foreach=True,
            )
        self.pool = nn.AvgPool1d(
            kernel_size=self._n_kernel_,
            stride=self._n_stride_,
            )
        self.schedule_cyclic = CyclicLR(
            self.optimizer,
            self._lr_min_,
            self._lr_max_,
            )
        self.schedule_warm = CosineAnnealingWarmRestarts(
            self.optimizer,
            self._warm_steps_,
            eta_min=self._lr_min_,
            )
        self.candles = None
        self.delta = None
        self.targets = None
        self.wax = None
        self.metrics = nn.ParameterDict()
        self.predictions = nn.ParameterDict()
        self.verbosity = int(verbosity)
        self.to(self._device_)
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set inputs to', self._n_inputs_)
            print(self._prefix_, 'set hidden to', self._n_hidden_)
            print(self._prefix_, 'set layers to', self._n_layers_)
            print(self._prefix_, 'set dropout to', self._n_dropout_)
            print(self._prefix_, 'set batch size to', self._batch_size_)
            print(self._prefix_, 'set kernel size to', self._n_kernel_)
            print(self._prefix_, 'set stride to', self._n_stride_)
            print(self._prefix_, 'set initial lr to', self._lr_init_)
            print(self._prefix_, 'set max lr to', self._lr_max_)
            print(self._prefix_, 'set min lr to', self._lr_min_)
            print(self._prefix_, 'set warm steps to', self._warm_steps_)
            print(self._prefix_, 'set cook time to', self._cook_time_)
            print('')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            state_path = f'{self._state_path_}/'
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
                state_path += self._symbol_
                self.candles = torch.load(
                    f'{state_path}.candles',
                    map_location=self._device_type_,
                    )
                self.delta = torch.load(
                    f'{state_path}.delta',
                    map_location=self._device_type_,
                    )
                self.targets = torch.load(
                    f'{state_path}.targets',
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
                state_path += self._symbol_
                torch.save(self.candles, f'{state_path}.candles')
                torch.save(self.delta, f'{state_path}.delta')
                torch.save(self.targets, f'{state_path}.targets')
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

    def __time_plot__(self, predictions, targets, adj_loss):
        epoch = self.metrics['epochs']
        x = range(predictions.shape[0])
        y_p = predictions.detach().cpu()
        y_t = targets.detach().cpu()
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

    def __time_step__(self, candles, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        if study:
            self.train()
            self.optimizer.zero_grad()
            coated_candles = self.forward(candles)
            return coated_candles.clone()
        else:
            self.eval()
            with torch.no_grad():
                coated_candles = self.forward(candles)
            return coated_candles.clone()

    def collect(self, offering):
        """Takes the offered dataframe and converts it to candle tensors."""
        n_time = len(offering.index)
        if n_time < self._batch_size_:
            return False
        params = self._p_tensor_
        tensor = torch.tensor
        candles = offering[self._candles_].to_numpy()
        candles = tensor(candles, **params)
        self.candles = candles.detach().cpu().requires_grad_(True)
        delta = offering[self._delta_].to_list()
        delta = tensor((delta,), **params).H
        self.delta = delta.detach().cpu()
        targets = offering[self._targets_].to_list()
        targets = tensor((targets,), **params).H
        self.targets = targets.detach().cpu()
        wax = offering[self._wax_].to_list()
        wax = tensor((wax,), **params).H
        self.wax = wax.detach().cpu()
        self.quick_save()
        return True

    def forward(self, candles):
        """**bubble*bubble**bubble**"""
        candles = self.normalizer(candles)
        candles = self.cauldron(candles)[1].H
        candles = self.pool(candles).H
        bubbles = torch.topk(candles.sum(1), self._batch_size_)
        candles = gelu(candles[bubbles.indices].sum(1)) / 3
        if self.verbosity > 1:
            print('bubbles:', bubbles.indices.tolist())
            print(
                'candles:',
                [float('{:.10f}'.format(i)) for i in candles.tolist()],
                )
        return candles.unsqueeze(0).H.clone()

    def predict(self, dataframe):
        """Take a batch of inputs and return the future signal."""
        self.quick_load()
        batch_size = self._batch_size_
        candles = self.candles[-batch_size:]
        wax = self.wax[-batch_size:]
        return self.__time_step__(candles, wax, study=False)

    def quick_load(self):
        """Alias to load RNN state."""
        self.__manage_state__(call_type=0)

    def quick_save(self):
        """Alias to save RNN state."""
        self.__manage_state__(call_type=1)

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        self.quick_load()
        batch_size = self._batch_size_
        batch_len = self.candles.shape[0]
        coated = batch_size
        if batch_len <= batch_size * 2:
            return False
        coating_candles = True
        cook_time = self._cook_time_
        warm_steps = self._warm_steps_
        if 'bubbling_wax' in self.metrics.keys():
            heating_up = False
        else:
            heating_up = True
        n_heat = 0
        n_heat_avg = inf
        n_reheat = 137
        sealed = dict(candles=list(), targets=list())
        self.metrics['epochs'] = 0
        self.metrics['loss'] = 0
        self.metrics['mae'] = 0
        self.metrics['mse'] = 0
        tolerance = self._tolerance_ * self.targets.mean().item()
        t_cook = time.time()
        vstack = torch.vstack
        while coating_candles:
            if batch_len <= coated + batch_size:
                if time.time() - t_cook >= cook_time:
                    coating_candles = False
                    epochs = self.metrics['epochs']
                    self.metrics['loss'] = self.metrics['loss'] / epochs
                    self.metrics['mae'] = self.metrics['mae'] / epochs
                    self.metrics['mse'] = sqrt(self.metrics['mse'])
                    break
                self.quick_save()
                c_plot = vstack(sealed['candles'])
                t_plot = vstack(sealed['targets'])
                self.__time_plot__(c_plot, t_plot, adj_loss)
                sealed = dict(candles=list(), targets=list())
                coated = batch_size
                continue
            c_i = coated - batch_size
            c_ii = coated + batch_size
            candles = self.candles[c_i:coated]
            targets = self.targets[coated:c_ii]
            t_delta = self.delta[coated:c_ii]
            wax = self.wax[c_i:coated]
            coated_candles = self.__time_step__(candles, study=True)
            loss = self.loss_fn(coated_candles, t_delta)
            loss.backward()
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
            adj_loss = sqrt(loss.item())
            n_heat += adj_loss
            coated_candles = wax + (wax * coated_candles)
            delta = (coated_candles - targets).abs()
            correct = delta[delta >= tolerance]
            correct = delta[delta <= tolerance].shape[0]
            absolute_error = batch_size - correct
            self.metrics['acc'] = (correct, batch_size)
            self.metrics['epochs'] += 1
            self.metrics['loss'] += adj_loss
            self.metrics['mae'] += absolute_error
            self.metrics['mse'] += absolute_error ** 2
            if coated % n_reheat == 0:
                n_heat = n_heat / n_reheat
                if n_heat >= n_heat_avg:
                    heating_up = True
                    n_heat_avg = inf
                else:
                    n_heat_avg = n_heat
                n_heat = 0
            sealed['candles'].append(coated_candles.clone())
            sealed['targets'].append(targets.clone())
            coated += batch_size
        self.quick_save()
        c_plot = vstack(sealed['candles'])
        t_plot = vstack(sealed['targets'])
        self.__time_plot__(c_plot, t_plot, adj_loss)
        if self.verbosity > 1: print('')
        return True
