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
from math import sqrt, inf
from os import mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
FEATURE_KEYS = [
    'open', 'high', 'low', 'close', 'volume', 'num_trades', 'vol_wma_price',
    'trend', 'fib_retrace_0.236', 'fib_retrace_0.382', 'fib_retrace_0.5',
    'fib_retrace_0.618', 'fib_retrace_0.786', 'fib_retrace_0.886',
    'fib_extend_0.236', 'fib_extend_0.382', 'fib_extend_0.5',
    'fib_extend_0.618', 'fib_extend_0.786', 'fib_extend_0.886',
    'price_zs', 'price_sdev', 'price_wema', 'price_dh', 'price_dl',
    'price_mid', 'volume_zs', 'volume_sdev', 'volume_wema',
    'volume_dh', 'volume_dl', 'volume_mid', 'delta',
    ]


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, *args, verbosity=0, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._batch_size_ = 8
        self._cook_time_ = 180
        self._c_iota_ = iota = 1 / 137
        self._c_phi_ = phi = 1.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._feature_keys_ = FEATURE_KEYS
        self._lr_init_ = phi - 1
        self._lr_max_ = iota / (phi - 1)
        self._lr_min_ = iota / phi
        self._n_cluster_ = 8
        self._n_dropout_ = ((137 * phi) ** iota) - 1
        self._n_hidden_ = 1024
        self._n_inputs_ = len(self._feature_keys_)
        self._n_layers_ = 16
        self._n_stride_ = 1
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self._targets_ = 'price_med'
        self._tolerance_ = 0.00033
        self._warm_steps_ = 3
        self.cauldron = nn.GRU(
            input_size=self._n_inputs_,
            hidden_size=self._n_hidden_,
            num_layers=self._n_layers_,
            bias=True,
            batch_first=True,
            dropout=self._n_dropout_,
            device=self._device_,
            )
        self.inscribe = nn.Linear(
            in_features=25,
            out_features=2,
            bias=True,
            **self._p_tensor_,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='mean',
            delta=1.0,
            )
        self.optimizer = RMSprop(
            self.parameters(),
            lr=self._lr_init_,
            foreach=True,
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
        self.stir = nn.Conv2d(
            in_channels=self._n_layers_,
            out_channels=self._n_layers_,
            kernel_size=self._n_cluster_,
            stride=self._n_stride_,
            )
        self.melt = nn.InstanceNorm1d(
            self._n_inputs_,
            momentum=0.1,
            **self._p_tensor_,
            )
        self.metrics = nn.ParameterDict({
            'acc': (0, 0),
            'bubbling_wax': True,
            'epochs': 0,
            'loss': 0,
            'mae': 0,
            'mse': 0,
            })
        self.predictions = nn.ParameterDict()
        self.verbosity = int(verbosity)
        self.to(self._device_)
        self.__manage_state__(call_type=0)
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

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        state_path = f'{self._state_path_}/MOIRAI.state'
        if call_type == 0:
            try:
                state = torch.load(state_path, map_location=self._device_type_)
                self.load_state_dict(state['moirai'])
                for key, value in state['metrics'].items():
                    self.metrics[key] = value
                for key, value in state['predictions'].items():
                    self.predictions[key] = value
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            except FileNotFoundError:
                self.__manage_state__(call_type=1)
            except Exception as details:
                if self.verbosity > 1:
                    print(self._prefix_, *details.args)
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

    def __time_plot__(self, predictions, targets, adj_loss):
        epoch = self.metrics['epochs']
        accuracy = self.metrics['acc']
        x = range(predictions.shape[0])
        y_p = predictions.detach().cpu()
        y_t = targets.detach().cpu()
        n_p = round(y_p[-1].item(), 2)
        n_t = round(y_t[-1].item(), 2)
        fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.3, 0.3, 0.3))
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.plot(x, y_t, label=f'Target: {n_t}', color='#FF9600')
        ax.plot(x, y_p, label=f'Prediction: {n_p}', color='#FFE88E')
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

    def forward(self, candles):
        """**bubble*bubble**bubble**"""
        coating = list()
        iota = self._c_iota_
        inscribe = self.inscribe
        n_batch = self._batch_size_
        n_dim = self._n_layers_ * 2
        bubbles = self.cauldron(self.melt(candles))[1]
        bubbles = bubbles.view(bubbles.shape[0], n_dim, n_dim)
        bubbles = self.stir(bubbles)
        for bubble in bubbles.split(1):
            bubble = bubble[0, bubble.sum(1).softmax(1).argmax(1), :]
            coating.append(inscribe(bubble))
        coating = torch.cat(coating) ** 2
        candles = list()
        for sigil in coating.chunk(n_batch, dim=0):
            candles.append(sigil.mean(1).mean(0))
        return torch.vstack(candles)

    def research(self, offering, dataframe):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        params = self._p_tensor_
        tensor = torch.tensor
        cheese_fresh = dataframe[self._feature_keys_].to_numpy()
        cheese_fresh = tensor(cheese_fresh, **params).requires_grad_(True)
        cheese_aged = dataframe[self._targets_].to_list()
        cheese_aged = tensor((cheese_aged,), **params).H
        batch_size = self._batch_size_
        batch_len = cheese_fresh.shape[0]
        coated = batch_size
        if batch_len <= batch_size * 2:
            return False
        coating_candles = True
        cook_time = self._cook_time_
        heating_up = self.metrics['bubbling_wax']
        n_heat = inf
        n_loss = 0
        sealed = dict(candles=list(), targets=list())
        self.metrics['acc'] = (0, 0)
        self.metrics['loss'] = 0
        self.metrics['mae'] = 0
        self.metrics['mse'] = 0
        tolerance = self._tolerance_ * cheese_aged.mean().item()
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
                self.metrics['epochs'] += 1
                epochs = self.metrics['epochs']
                if epochs % warm_steps == 0:
                    n_loss = n_loss / warm_steps
                    if n_loss >= n_heat:
                        heating_up = self.metrics['bubbling_wax'] = True
                        n_heat = inf
                        if self.verbosity > 1:
                            print('heating up!')
                    else:
                        heating_up = self.metrics['bubbling_wax'] = False
                        n_heat = n_loss
                        if self.verbosity > 1:
                            print('cooking...')
                    n_loss = 0
                if time.time() - t_cook < cook_time:
                    coated = batch_size
                    n_heat = inf
                    n_loss = 0
                    sealed = dict(candles=list(), targets=list())
                    self.metrics['acc'] = (0, 0)
                    self.metrics['loss'] = 0
                    self.metrics['mae'] = 0
                    self.metrics['mse'] = 0
                else:
                    coating_candles = False
                    self.__manage_state__(call_type=1)
                continue
            candles = cheese_fresh[c_i:coated]
            targets = cheese_aged[coated:c_ii]
            coated_candles = self.__time_step__(candles, study=True)
            loss = self.loss_fn(coated_candles, targets)
            loss.backward()
            self.optimizer.step()
            if not heating_up:
                self.schedule_cyclic.step()
            else:
                self.schedule_warm.step()
            adj_loss = sqrt(loss.item()) / coated
            delta = (coated_candles - targets).abs()
            correct = delta[delta >= tolerance]
            correct = delta[delta <= tolerance].shape[0]
            absolute_error = batch_size - correct
            t_correct = correct + self.metrics['acc'][0]
            t_batch = batch_size + self.metrics['acc'][1]
            self.metrics['acc'] = (t_correct, t_batch)
            self.metrics['loss'] += adj_loss
            self.metrics['mae'] += absolute_error
            self.metrics['mse'] += absolute_error ** 2
            sealed['candles'].append(coated_candles.clone())
            sealed['targets'].append(targets.clone())
            n_loss += adj_loss
            coated += batch_size
        self.__time_plot__(
            vstack(sealed['candles']),
            vstack(sealed['targets']),
            self.metrics['loss'],
            )
        return self.__time_step__(candles, study=False).clone()
