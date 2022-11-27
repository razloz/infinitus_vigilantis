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
        self._cook_time_ = 3
        self._c_iota_ = iota = 1 / 137
        self._c_phi_ = phi = 1.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        if not exists(abspath('./rnn')): mkdir(abspath('./rnn'))
        self._state_path_ = abspath('./rnn/states')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._feature_keys_ = FEATURE_KEYS
        self._lr_init_ = phi - 1
        self._lr_max_ = iota / (phi - 1)
        self._lr_min_ = iota / phi
        self._n_cluster_ = 3
        self._n_dim_ = 6
        self._n_features_ = len(self._feature_keys_)
        self._n_hidden_ = self._n_dim_ ** 3
        self._n_inputs_ = len(self._feature_keys_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self._targets_ = 'price_med'
        self._tolerance_ = 0.00033
        self._warm_steps_ = 3
        self.cauldron = nn.GRU(
            input_size=self._n_inputs_,
            hidden_size=self._n_hidden_,
            bias=True,
            batch_first=True,
            device=self._device_,
            )
        self.inscription = nn.Linear(
            in_features=4 ** 3,
            out_features=1,
            bias=True,
            **self._p_tensor_,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='mean',
            delta=1.0,
            )
        self.melt = nn.InstanceNorm1d(
            self._n_inputs_,
            momentum=0.1,
            **self._p_tensor_,
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
        self.stir = nn.Conv3d(
            in_channels=self._n_hidden_,
            out_channels=self._n_hidden_,
            kernel_size=self._n_cluster_,
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
        if self.verbosity > 0:
            print(self._prefix_, 'set device type to', self._device_type_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set inputs to', self._n_inputs_)
            print(self._prefix_, 'set dim shape to', self._n_dim_)
            print(self._prefix_, 'set hidden to', self._n_hidden_)
            print(self._prefix_, 'set cluster size to', self._n_cluster_)
            print(self._prefix_, 'set initial lr to', self._lr_init_)
            print(self._prefix_, 'set max lr to', self._lr_max_)
            print(self._prefix_, 'set min lr to', self._lr_min_)
            print(self._prefix_, 'set warm steps to', self._warm_steps_)
            print(self._prefix_, 'set cook time to', self._cook_time_)
            print('')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        state_path = f'{self._state_path_}/{self._symbol_}.state'
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
        y_p = predictions.squeeze(1).tolist()
        y_t = targets.squeeze(1).tolist()
        n_p = round(y_p[-1], 2)
        n_t = round(y_t[-1], 2)
        fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.3, 0.3, 0.3))
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.plot(x, y_p, label=f'Prediction: {n_p}', color='#FFE88E')
        ax.plot(x, y_t, label=f'Target: {n_t}', color='#FF9600')
        symbol = self._symbol_
        title = f'{symbol}\n'
        title += f'Accuracy: {accuracy}, '
        title += f'Epochs: {epoch}, '
        title += 'Loss: {:.10f}'.format(adj_loss)
        fig.suptitle(title, fontsize=18)
        plt.legend(fancybox=True, loc='best', ncol=1)
        plt.savefig(f'{self._epochs_path_}/{symbol}.png')
        plt.clf()
        plt.close()
        if self.verbosity > 0:
            prefix = self._prefix_
            print(prefix, 'PREDICTIONS TAIL:', y_p[-1])
            print(prefix, 'TARGET TAIL:', y_t[-1])
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
        inscription = self.inscription
        n_dim = self._n_dim_
        bubbles = self.cauldron(self.melt(candles))[0]
        bubbles = bubbles.view(bubbles.shape[0], n_dim, n_dim, n_dim)
        bubbles = self.stir(bubbles)
        sigil = list()
        for bubble in bubbles.split(1):
            sigil.append(inscription(bubble.flatten(0)))
        candles = torch.vstack(sigil) ** 2
        return candles

    def research(self, offering, dataframe):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        self.__manage_state__(call_type=0)
        params = self._p_tensor_
        tensor = torch.tensor
        cheese_fresh = dataframe[self._feature_keys_].to_numpy()
        cheese_fresh = tensor(cheese_fresh, **params).requires_grad_(True)
        cheese_aged = dataframe[self._targets_].to_list()
        cheese_aged = tensor((cheese_aged,), **params).H
        batch_size = cheese_fresh.shape[0]
        if batch_size % 2 != 0:
            batch_size -= 1
            cheese_aged = cheese_aged[1:]
            cheese_fresh = cheese_fresh[1:]
        batch_size = int(batch_size / 2)
        candles = cheese_fresh[:-batch_size]
        targets = cheese_aged[batch_size:]
        coating_candles = True
        cook_time = self._cook_time_
        epochs = 0
        heating_up = self.metrics['bubbling_wax']
        n_heat = inf
        n_loss = 0
        self.metrics['acc'] = (0, 0)
        self.metrics['loss'] = 0
        self.metrics['mae'] = 0
        self.metrics['mse'] = 0
        tolerance = self._tolerance_ * cheese_aged.mean().item()
        vstack = torch.vstack
        warm_steps = self._warm_steps_
        self.stir = nn.Conv3d(
            in_channels=batch_size,
            out_channels=batch_size,
            kernel_size=self._n_cluster_,
            **self._p_tensor_,
            )
        t_cook = time.time()
        while coating_candles:
            coated_candles = self.__time_step__(candles, study=True)
            loss = self.loss_fn(coated_candles, targets)
            loss.backward()
            self.optimizer.step()
            if not heating_up:
                self.schedule_cyclic.step()
            else:
                self.schedule_warm.step()
            adj_loss = sqrt(loss.item()) / batch_size
            delta = (coated_candles - targets).abs()
            correct = delta[delta >= tolerance]
            correct = delta[delta <= tolerance].shape[0]
            absolute_error = batch_size - correct
            self.metrics['acc'] = (correct, batch_size)
            self.metrics['loss'] += adj_loss
            self.metrics['mae'] += absolute_error
            self.metrics['mse'] += absolute_error ** 2
            self.metrics['epochs'] += 1
            epochs += 1
            n_loss += adj_loss
            if self.metrics['epochs'] % warm_steps == 0:
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
            if time.time() - t_cook >= cook_time:
                coating_candles = False
                self.__manage_state__(call_type=1)
        self.metrics['loss'] = self.metrics['loss'] / epochs
        self.metrics['mae'] = self.metrics['mae'] / epochs
        self.metrics['mse'] = sqrt(self.metrics['mse']) / epochs
        self.__time_plot__(coated_candles, targets, self.metrics['loss'])
        #candles = cheese_fresh[batch_size:]
        return True #self.__time_step__(candles, study=False).clone()
