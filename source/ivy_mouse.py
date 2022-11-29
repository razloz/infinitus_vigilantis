"""Three blind mice to predict the future."""
import time
import torch
import traceback
import matplotlib.pyplot as plt
import source.ivy_commons as icy
from torch import nn
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, TensorDataset
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
        self._cook_time_ = 45
        self._c_iota_ = iota = 1 / 137
        self._c_phi_ = phi = 1.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._feature_keys_ = FEATURE_KEYS
        self._lr_init_ = iota ** (phi - 1)
        self._lr_max_ = iota / (phi - 1)
        self._lr_min_ = iota / phi
        self._n_batch_ = b_size = 34
        self._n_cluster_ = 3
        self._n_dropout_ = iota
        self._n_dim_ = n_dim = 9
        self._n_features_ = len(self._feature_keys_)
        self._n_hidden_ = n_dim ** 3
        self._n_inputs_ = self._n_features_
        self._n_layers_ = n_dim
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self._targets_ = 'price_med'
        self._tolerance_ = iota ** phi
        self._warm_steps_ = 34
        self.cauldron = nn.GRU(
            input_size=self._n_inputs_,
            hidden_size=self._n_hidden_,
            num_layers=self._n_layers_,
            dropout=self._n_dropout_,
            bias=True,
            batch_first=True,
            device=self._device_,
            )
        self.inscription = nn.Linear(
            in_features=b_size,
            out_features=b_size,
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
            in_channels=n_dim,
            out_channels=n_dim,
            kernel_size=self._n_cluster_,
            **self._p_tensor_,
            )
        self.metrics = nn.ParameterDict({
            'acc': [0, 0],
            'bubbling_wax': True,
            'epochs': 0,
            'loss': 0,
            'mae': 0,
            'mse': 0,
            })
        self.verbosity = int(verbosity)
        self.to(self._device_)
        self.__manage_state__(call_type=0)
        if self.verbosity > 0:
            print(self._prefix_, 'set device type to', self._device_type_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set inputs to', self._n_inputs_)
            print(self._prefix_, 'set batch size to', self._n_batch_)
            print(self._prefix_, 'set dim shape to', self._n_dim_)
            print(self._prefix_, 'set hidden to', self._n_hidden_)
            print(self._prefix_, 'set layers to', self._n_layers_)
            print(self._prefix_, 'set dropout to', self._n_dropout_)
            print(self._prefix_, 'set cluster size to', self._n_cluster_)
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
                    },
                state_path,
                )
            if self.verbosity > 2:
                print(self._prefix_, 'Saved RNN state.')

    def __time_plot__(self, predictions, targets, adj_loss):
        fig = plt.figure(figsize=(6.40, 4.80), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.3, 0.3, 0.3))
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        epoch = self.metrics['epochs']
        accuracy = self.metrics['acc']
        x = range(predictions.shape[0])
        y_p = predictions.squeeze(1).tolist()
        y_t = targets.squeeze(1).tolist()
        n_p = round(y_p[-1], 2)
        n_t = round(y_t[-1], 2)
        ax.plot(x, y_p, label=f'Prediction: {n_p}', color='#FFE88E')
        ax.plot(x, y_t, label=f'Target: {n_t}', color='#FF9600')
        symbol = self._symbol_
        title = f'{symbol}\n'
        title += f'Accuracy: {accuracy}, '
        title += f'Epochs: {epoch}'
        fig.suptitle(title, fontsize=18)
        plt.legend(fancybox=True, loc='best', ncol=1)
        plt.savefig(f'{self._epochs_path_}/{int(time.time())}.png')
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
            return self.forward(candles)
        else:
            self.eval()
            with torch.no_grad():
                return self.forward(candles)

    def forward(self, candles):
        """**bubble*bubble**bubble**"""
        n_dim = self._n_dim_
        bubbles = self.cauldron(self.melt(candles))[1]
        bubbles = bubbles.view(n_dim, n_dim, n_dim, n_dim)
        bubbles = self.stir(bubbles).flatten(0)
        sigil = torch.topk(bubbles, self._n_batch_)[0]
        candles = self.inscription(sigil) ** 2
        return candles.unsqueeze(0).H

    def research(self, offering, dataframe):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        batch = self._n_batch_
        cook_time = self._cook_time_
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        p_tensor = self._p_tensor_
        research = self.__time_step__
        tensor = torch.tensor
        vstack = torch.vstack
        warm_steps = self._warm_steps_
        trim = 0
        data_len = len(dataframe)
        while data_len % batch != 0:
            dataframe = dataframe[1:]
            data_len = len(dataframe)
        cheese_fresh = dataframe[self._feature_keys_].to_numpy()
        cheese_fresh = tensor(cheese_fresh, requires_grad=True, **p_tensor)
        cheese_aged = dataframe[self._targets_].to_list()
        cheese_aged = tensor((cheese_aged,), **p_tensor).H
        sample = TensorDataset(cheese_fresh[:-batch], cheese_aged[batch:])
        candles = DataLoader(sample, batch_size=batch)
        tolerance = self._tolerance_ * cheese_aged.mean(0).item()
        self.metrics['loss'] = 0
        self.metrics['mae'] = 0
        self.metrics['mse'] = 0
        epochs = 0
        n_loss = 0
        n_heat = inf
        heating_up = self.metrics['bubbling_wax']
        t_cook = time.time()
        cooking = True
        while cooking:
            self.metrics['acc'] = [0, 0]
            predictions = list()
            for features, targets in iter(candles):
                coated_candles = research(features, study=True)
                predictions.append(coated_candles)
                loss = loss_fn(coated_candles, targets)
                loss.backward()
                optimizer.step()
                adj_loss = sqrt(loss.item()) / batch
                n_loss += adj_loss
                delta = (coated_candles - targets).abs()
                correct = delta[delta >= tolerance]
                correct = delta[delta <= tolerance].shape[0]
                absolute_error = batch - correct
                self.metrics['acc'][0] += correct
                self.metrics['acc'][1] += batch
                self.metrics['loss'] += adj_loss
                self.metrics['mae'] += absolute_error
                self.metrics['mse'] += absolute_error ** 2
            if not heating_up:
                self.schedule_cyclic.step()
            else:
                self.schedule_warm.step()
            self.metrics['epochs'] += 1
            epochs += 1
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
                cooking = False
        self.metrics['loss'] = self.metrics['loss'] / epochs
        self.metrics['mae'] = self.metrics['mae'] / epochs
        self.metrics['mse'] = sqrt(self.metrics['mse']) / epochs
        self.__time_plot__(
            vstack(predictions),
            cheese_aged[batch:],
            self.metrics['loss'],
            )
        self.__manage_state__(call_type=1)
        #candles = cheese_fresh[batch:]
        return True #self.__time_step__(candles, study=False).clone()
