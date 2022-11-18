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
    def __init__(self, *args, verbosity=0, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._batch_size_ = 34
        self._candles_ = ['open', 'high', 'low', 'close']
        self._cook_time_ = 3600
        self._c_iota_ = iota = 1 / 137
        self._c_phi_ = phi = 1.618033988749894
        self._delta_ = 'delta'
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._offerings_path_ = abspath('./rnn/offerings')
        if not exists(self._offerings_path_): mkdir(self._offerings_path_)
        self._lr_init_ = phi
        self._lr_max_ = iota / (phi - 1)
        self._lr_min_ = iota / phi
        self._n_cluster_ = 9
        self._n_dropout_ = ((137 * phi) ** iota) - 1
        self._n_hidden_ = 207
        self._n_inputs_ = len(self._candles_)
        self._n_layers_ = 207
        self._n_stride_ = 3
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self._targets_ = 'price_med'
        self._tolerance_ = 0.00033
        self._warm_steps_ = 3
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
            reduction='mean',
            delta=1.0,
            )
        self.normalizer = nn.InstanceNorm1d(
            self._n_inputs_,
            eps=iota,
            momentum=0.1,
            **self._p_tensor_,
            )
        self.optimizer = RMSprop(
            self.cauldron.parameters(),
            lr=self._lr_init_,
            foreach=True,
            )
        self.pool = nn.AvgPool1d(
            kernel_size=self._n_cluster_,
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
        self.metrics = nn.ParameterDict({
            'acc': (0, 0),
            'bubbling_wax': True,
            'epochs': 0,
            'loss': 0,
            'mae': 0,
            'mse': 0,
            })
        self.offerings = nn.ParameterDict()
        self.predictions = nn.ParameterDict()
        self.verbosity = int(verbosity)
        self.to(self._device_)
        if self.verbosity > 0:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set inputs to', self._n_inputs_)
            print(self._prefix_, 'set hidden to', self._n_hidden_)
            print(self._prefix_, 'set layers to', self._n_layers_)
            print(self._prefix_, 'set dropout to', self._n_dropout_)
            print(self._prefix_, 'set batch size to', self._batch_size_)
            print(self._prefix_, 'set cluster size to', self._n_cluster_)
            print(self._prefix_, 'set stride to', self._n_stride_)
            print(self._prefix_, 'set initial lr to', self._lr_init_)
            print(self._prefix_, 'set max lr to', self._lr_max_)
            print(self._prefix_, 'set min lr to', self._lr_min_)
            print(self._prefix_, 'set warm steps to', self._warm_steps_)
            print(self._prefix_, 'set cook time to', self._cook_time_)
            print('')

    def __manage_state__(self, symbol, call_type=0):
        """Handles loading and saving of the RNN state."""
        state_path = f'{self._state_path_}/{symbol}.state'
        if call_type == 0:
            state = torch.load(state_path, map_location=self._device_type_)
            self.load_state_dict(state['moirai'])
            for key, value in state['metrics'].items():
                self.metrics[key] = value
            for key, value in state['predictions'].items():
                self.predictions[key] = value
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

    def __manage_tensors__(self, offerings, call_type=0):
        """Handles loading and saving of the offering tensors."""
        if call_type == 0:
            for symbol in offerings:
                symbol = symbol.upper()
                state_path = f'{self._offerings_path_}/{symbol}'
                self.offerings[symbol] = dict()
                self.offerings[symbol]['candles'] = torch.load(
                    f'{state_path}.candles',
                    map_location=self._device_type_,
                    )
                self.offerings[symbol]['delta'] = torch.load(
                    f'{state_path}.delta',
                    map_location=self._device_type_,
                    )
                self.offerings[symbol]['targets'] = torch.load(
                    f'{state_path}.targets',
                    map_location=self._device_type_,
                    )
                self.offerings[symbol]['wax'] = torch.load(
                    f'{state_path}.wax',
                    map_location=self._device_type_,
                    )
            if self.verbosity > 2:
                print(self._prefix_, f'Loaded {len(offerings)} offerings.')
        elif call_type == 1:
            for symbol in offerings:
                if len(self.offerings[symbol].keys()) == 0:
                    continue
                symbol = symbol.upper()
                state_path = f'{self._offerings_path_}/{symbol}'
                torch.save(
                    self.offerings[symbol]['candles'],
                    f'{state_path}.candles',
                    )
                torch.save(
                    self.offerings[symbol]['delta'],
                    f'{state_path}.delta',
                    )
                torch.save(
                    self.offerings[symbol]['targets'],
                    f'{state_path}.targets',
                    )
                torch.save(
                    self.offerings[symbol]['wax'],
                    f'{state_path}.wax',
                    )
            if self.verbosity > 2:
                print(self._prefix_, f'Saved {len(offerings)} offerings.')

    def __time_plot__(self, predictions, targets, wax, adj_loss):
        epoch = self.metrics['epochs']
        accuracy = self.metrics['acc']
        x = range(predictions.shape[0])
        y_p = predictions.detach().cpu()
        y_t = targets.detach().cpu()
        y_w = wax.detach().cpu()
        n_p = round(y_p[-1].item(), 2)
        n_t = round(y_t[-1].item(), 2)
        n_w = round(y_w[-1].item(), 2)
        fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.3, 0.3, 0.3))
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.plot(x, y_t, label=f'Target: {n_t}', color='#FF9600')
        ax.plot(x, y_p, label=f'Prediction: {n_p}', color='#FFE88E')
        ax.plot(x, y_w, label=f'Wax: {n_w}', color='#FFFFE0', linestyle=':')
        title = f'{self._symbol_}\n'
        title += f'Accuracy: {accuracy}, '
        title += f'Epochs: {epoch}, '
        title += 'Loss: {:.10f}'.format(adj_loss)
        fig.suptitle(title, fontsize=18)
        plt.legend(fancybox=True, loc='best', ncol=1)
        ts = int(time.time())
        epochs_path = f'{self._epochs_path_}/{ts}.png'
        plt.savefig(epochs_path)
        plt.clf()
        plt.close()
        if self.verbosity > 0:
            prefix = self._prefix_
            print(prefix, 'PREDICTIONS TAIL:', y_p[-1].item())
            print(prefix, 'TARGET TAIL:', y_t[-1].item())
            print(prefix, 'WAX TAIL:', y_w[-1].item())
            print(prefix, 'ADJ_LOSS:', adj_loss)
            print(prefix, 'ACC:', accuracy)
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

    def collect(self, offering, candles):
        """Takes the offered dataframe and converts it to candle tensors."""
        n_time = len(candles.index)
        if n_time < self._batch_size_:
            return False
        symbol = self._symbol_ = str(offering).upper()
        params = self._p_tensor_
        tensor = torch.tensor
        cdls = candles[self._candles_].to_numpy()
        cdls = tensor(cdls, **params).requires_grad_(True)
        delta = candles[self._delta_].to_list()
        delta = tensor((delta,), **params).H
        targets = candles[self._targets_].to_list()
        targets = tensor((targets,), **params).H
        wax = candles[self._wax_].to_list()
        wax = tensor((wax,), **params).H
        self.offerings[symbol] = dict()
        self.offerings[symbol]['candles'] = cdls.detach().cpu()
        self.offerings[symbol]['delta'] = delta.detach().cpu()
        self.offerings[symbol]['targets'] = targets.detach().cpu()
        self.offerings[symbol]['wax'] = wax.detach().cpu()
        return True

    def forward(self, candles):
        """**bubble*bubble**bubble**"""
        candles = self.normalizer(candles)
        candles = self.cauldron(candles)[1]
        candles = self.pool(candles)
        coating = not bool(self.metrics['bubbling_wax'])
        bubbles = torch.topk(candles, 3, largest=coating)
        candles = candles[:, bubbles.indices[1]].mean(1)
        coating = not coating
        bubbles = torch.topk(candles, self._batch_size_, largest=coating)
        candles = candles[bubbles.indices]
        if self.verbosity > 1:
            print('bubbles:', bubbles.indices.tolist())
            print(
                'candles:',
                [float('{:.10f}'.format(i)) for i in candles.tolist()],
                )
        return candles.unsqueeze(0).H.clone()

    def predict(self, offering):
        """Take a batch of inputs and return the future signal."""
        symbol = self._symbol_ = str(offering).upper()
        candles = self.offerings[symbol]['candles']
        wax = self.offerings[symbol]['wax']
        batch_size = self._batch_size_
        candles = candles[-batch_size:]
        wax = wax[-batch_size:]
        return self.__time_step__(candles, wax, study=False)

    def quick_load(self, rnn=True, offerings=None):
        """Alias to load RNN state and/or offering tensors."""
        if rnn:
            for symbol in offerings:
                self.__manage_state__(symbol, call_type=0)
        if type(offerings) == list:
            self.__manage_tensors__(offerings, call_type=0)


    def quick_save(self, rnn=True, offerings=None):
        """Alias to save RNN state and/or offering tensors."""
        if rnn:
            for symbol in offerings:
                self.__manage_state__(symbol, call_type=1)
        if type(offerings) == list:
            self.__manage_tensors__(offerings, call_type=1)

    def research(self, offering):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        all_candles = self.offerings[symbol]['candles']
        all_targets = self.offerings[symbol]['targets']
        all_delta = self.offerings[symbol]['delta']
        all_wax = self.offerings[symbol]['wax']
        batch_size = self._batch_size_
        batch_len = all_candles.shape[0]
        coated = batch_size
        if batch_len <= batch_size * 2:
            return False
        coating_candles = True
        cook_time = self._cook_time_
        heating_up = self.metrics['bubbling_wax']
        n_heat = inf
        n_loss = 0
        sealed = dict(candles=list(), targets=list(), wax=list())
        self.metrics['acc'] = (0, 0)
        self.metrics['loss'] = 0
        self.metrics['mae'] = 0
        self.metrics['mse'] = 0
        tolerance = self._tolerance_ * all_targets.mean().item()
        t_cook = time.time()
        vstack = torch.vstack
        warm_steps = self._warm_steps_
        while coating_candles:
            c_i = coated - batch_size
            c_ii = coated + batch_size
            if batch_len <= c_ii:
                self.metrics['loss'] = self.metrics['loss'] / coated
                self.metrics['mae'] = self.metrics['mae'] / coated
                self.metrics['mse'] = sqrt(self.metrics['mse']) / coated
                c_plot = vstack(sealed['candles'])
                t_plot = vstack(sealed['targets'])
                w_plot = vstack(sealed['wax'])
                self.predictions[symbol] = dict(self.metrics)
                self.__time_plot__(c_plot, t_plot, w_plot, adj_loss)
                if time.time() - t_cook < cook_time:
                    coated = batch_size
                    n_heat = inf
                    n_loss = 0
                    sealed = dict(candles=list(), targets=list(), wax=list())
                    self.metrics['acc'] = (0, 0)
                    self.metrics['loss'] = 0
                    self.metrics['mae'] = 0
                    self.metrics['mse'] = 0
                else:
                    coating_candles = False
                continue
            candles = all_candles[c_i:coated]
            targets = all_targets[coated:c_ii]
            t_delta = all_delta[coated:c_ii]
            wax = all_wax[c_i:coated]
            coated_candles = self.__time_step__(candles, study=True)
            loss = self.loss_fn(coated_candles, t_delta)
            loss.backward()
            self.optimizer.step()
            if not heating_up:
                self.schedule_cyclic.step()
            else:
                self.schedule_warm.step()
            adj_loss = sqrt(loss.item())
            coated_candles = wax + (wax * coated_candles)
            delta = (coated_candles - targets).abs()
            correct = delta[delta >= tolerance]
            correct = delta[delta <= tolerance].shape[0]
            absolute_error = batch_size - correct
            t_correct = correct + self.metrics['acc'][0]
            t_batch = batch_size + self.metrics['acc'][1]
            self.metrics['acc'] = (t_correct, t_batch)
            self.metrics['epochs'] += 1
            self.metrics['loss'] += adj_loss
            self.metrics['mae'] += absolute_error
            self.metrics['mse'] += absolute_error ** 2
            sealed['candles'].append(coated_candles.clone())
            sealed['targets'].append(targets.clone())
            sealed['wax'].append(wax.clone())
            n_loss += adj_loss
            if self.metrics['epochs'] % warm_steps == 0:
                n_loss = n_loss / warm_steps
                if n_loss >= n_heat:
                    heating_up = self.metrics['bubbling_wax'] = True
                    n_heat = inf
                    print('heating up!')
                else:
                    heating_up = self.metrics['bubbling_wax'] = False
                    n_heat = n_loss
                    print('cooking...')
                n_loss = 0
            coated += batch_size
        return True
