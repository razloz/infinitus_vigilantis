"""Three blind mice to predict the future."""
import time
import torch
import traceback
import matplotlib.pyplot as plt
import numpy as np
import source.ivy_commons as icy
from torch import nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt, inf
from os import mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
FEATURE_KEYS = [
    'volume', 'num_trades', 'vol_wma_price', 'trend',
    'fib_retrace_0.236', 'fib_retrace_0.382', 'fib_retrace_0.5',
    'fib_retrace_0.618', 'fib_retrace_0.786', 'fib_retrace_0.886',
    'fib_extend_0.236', 'fib_extend_0.382', 'fib_extend_0.5',
    'fib_extend_0.618', 'fib_extend_0.786', 'fib_extend_0.886',
    'price_zs', 'price_sdev', 'price_wema', 'price_dh', 'price_dl',
    'price_mid', 'volume_zs', 'volume_sdev', 'volume_wema',
    'volume_dh', 'volume_dl', 'volume_mid',
    ]
TARGET_KEYS = [
    'open', 'high', 'low', 'close',
    ]


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, *args, batch_size=5, cook_time=0, verbosity=0, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        iota = 1 / 137
        phi = 1.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._feature_keys_ = FEATURE_KEYS
        self._target_keys_ = TARGET_KEYS
        self._constants_ = constants = {
            'activations': 18,
            'batch_size': batch_size,
            'cluster_shape': 3,
            'cook_time': cook_time,
            'dim': 9,
            'dropout': iota,
            'eps': iota * 1e-6,
            'features': len(self._feature_keys_),
            'hidden': 9 ** 3,
            'layers': 18,
            'lr_decay': iota / (phi - 1),
            'lr_init': iota ** (phi - 1),
            'momentum': phi * (phi - 1),
            'tolerance': iota ** phi,
            'weight_decay': iota / phi,
            }
        self._prefix_ = prefix = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self.cauldron = nn.GRU(
            input_size=constants['features'],
            hidden_size=constants['hidden'],
            num_layers=constants['layers'],
            dropout=constants['dropout'],
            bias=True,
            batch_first=True,
            device=self._device_,
            )
        self.inscription = nn.Linear(
            in_features=constants['activations'],
            out_features=2,
            bias=True,
            **self._p_tensor_,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='sum',
            delta=phi,
            )
        self.melt = nn.InstanceNorm1d(
            constants['features'],
            momentum=constants['momentum'],
            eps=constants['eps'],
            **self._p_tensor_,
            )
        self.optimizer = Adagrad(
            self.parameters(),
            lr=constants['lr_init'],
            lr_decay=constants['lr_decay'],
            weight_decay=constants['weight_decay'],
            eps=constants['eps'],
            foreach=True,
            maximize=False,
            )
        self.stir = nn.Conv3d(
            in_channels=constants['layers'],
            out_channels=constants['layers'],
            kernel_size=constants['cluster_shape'],
            **self._p_tensor_,
            )
        self.metrics = nn.ParameterDict({
            'acc': [0, 0],
            'epochs': 0,
            'loss': 0,
            'mae': 0,
            'mse': 0,
            'predictions': list(),
            })
        self.verbosity = int(verbosity)
        self.to(self._device_)
        if self.verbosity > 1:
            for key, value in constants.items():
                print(prefix, f'set {key} to {value}')
            print('')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        state_path = f'{self._state_path_}/.MOIRAI.state'
        if call_type == 0:
            try:
                state = torch.load(state_path, map_location=self._device_type_)
                self.load_state_dict(state['moirai'])
                for key, value in state['metrics'].items():
                    self.metrics[key] = value
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            except FileNotFoundError:
                self.metrics['epochs'] = 0
                self.metrics['predictions'] = list()
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

    def __time_plot__(self, predictions, targets):
        fig = plt.figure(figsize=(10.24, 7.68), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.3, 0.3, 0.3))
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_xlabel('Epoch', fontweight='bold')
        accuracy = self.metrics['acc']
        adj_loss = self.metrics['loss']
        batch_size = self._constants_['batch_size']
        epoch = self.metrics['epochs']
        x = range(len(predictions))
        y_p = predictions
        y_t = targets.mean(1).detach().cpu().tolist()
        predictions_tail = y_p[-1]
        target_tail = y_t[-1]
        y_t += [None for _ in range(batch_size)]
        ax.plot(x, y_p, color='#FFE88E') #gouda
        ax.plot(x, y_t, color='#FF9600') #cheddar
        symbol = self._symbol_
        title = f'{symbol}\n'
        title += f'Accuracy: {accuracy}, '
        title += f'Epochs: {epoch}, '
        title += f'Batch Size: {batch_size}'
        fig.suptitle(title, fontsize=18)
        file_name = f'{symbol}-{x[-1] - batch_size}-{int(time.time())}'
        plt.savefig(f'{self._epochs_path_}/{file_name}.png')
        plt.clf()
        plt.close()
        if self.verbosity > 0:
            prefix = self._prefix_
            print(prefix, 'PREDICTIONS TAIL:', predictions_tail)
            print(prefix, 'TARGET TAIL:', target_tail)
            print(prefix, 'ADJ_LOSS:', adj_loss)
            print(prefix, 'ACC:', accuracy)
            print(prefix, 'BATCH_SIZE:', batch_size)
            for k in ['loss', 'mae', 'mse']:
                v = self.metrics[k]
                if k == 'mse':
                    print(prefix, f'{k.upper()}: {sqrt(v)}')
                else:
                    print(prefix, f'{k.upper()}: {v / epoch}')
            print(prefix, 'EPOCH:', epoch)
            print('')

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
        constants = self._constants_
        dim = constants['dim']
        bubbles = self.cauldron(self.melt(candles))[1]
        bubbles = bubbles.view(bubbles.shape[0], dim, dim, dim)
        bubbles = self.stir(bubbles).flatten(0)
        sigil = torch.topk(bubbles.softmax(0), constants['activations'])[1]
        return self.inscription(bubbles[sigil]) ** 2

    def research(self, offering, dataframe):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        constants = self._constants_
        batch = constants['batch_size']
        batch_range = range(batch - 2)
        cook_time = constants['cook_time']
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        p_tensor = self._p_tensor_
        research = self.__time_step__
        tensor = torch.tensor
        hstack = torch.hstack
        vstack = torch.vstack
        data_len = len(dataframe)
        if data_len >= batch * 2:
            self.__manage_state__(call_type=0)
        else:
            return False
        while data_len % batch != 0:
            dataframe = dataframe[1:]
            data_len = len(dataframe)
        cheese_fresh = dataframe[self._feature_keys_].to_numpy()
        cheese_fresh = tensor(cheese_fresh, requires_grad=True, **p_tensor)
        cheese_aged = dataframe[self._target_keys_].to_numpy()
        cheese_aged = tensor(cheese_aged, **p_tensor)
        sample = TensorDataset(cheese_fresh[:-batch], cheese_aged[batch:])
        candles = DataLoader(sample, batch_size=batch)
        mean_cheese = cheese_aged.flatten(0).mean(0)
        tolerance = constants['tolerance'] * mean_cheese
        n_targets = batch * cheese_aged.shape[1]
        self.metrics['loss'] = 0
        self.metrics['mae'] = 0
        self.metrics['mse'] = 0
        epochs = 0
        cooking = True
        t_cook = time.time()
        null_pad = [None for _ in batch_range]
        while cooking:
            self.metrics['acc'] = [0, 0]
            predictions = [None, None]
            predictions += null_pad
            for features, targets in iter(candles):
                cdls = research(features, study=True)
                cdls = hstack([
                    cdls[0].unsqueeze(0),
                    cdls[-1].unsqueeze(0),
                    ])
                targets = targets.mean(1)
                targets = hstack([
                    targets[0].unsqueeze(0),
                    targets[-1].unsqueeze(0),
                    ])
                loss = loss_fn(cdls, targets)
                loss.backward()
                optimizer.step()
                cdl_open = float(cdls[0])
                cdl_close = float(cdls[-1])
                cdl_pad = (cdl_close - cdl_open) / (batch - 2)
                predictions.append(cdl_open)
                predictions += [
                    cdl_open + (cdl_pad * (n + 1)) for n in batch_range
                    ]
                predictions.append(cdl_close)
                adj_loss = sqrt(loss.item())
                adj_loss = 1 + (adj_loss - mean_cheese) / mean_cheese
                delta = (cdls - targets).abs()
                correct = delta[delta >= tolerance]
                correct = delta[delta <= tolerance].shape[0]
                absolute_error = n_targets - correct
                self.metrics['acc'][0] += correct
                self.metrics['acc'][1] += 2
                self.metrics['loss'] += adj_loss.item()
                self.metrics['mae'] += absolute_error
                self.metrics['mse'] += absolute_error ** 2
            self.metrics['epochs'] += 1
            epochs += 1
            if time.time() - t_cook >= cook_time:
                cooking = False
        self.metrics['loss'] = self.metrics['loss'] / epochs
        self.metrics['mae'] = self.metrics['mae'] / epochs
        self.metrics['mse'] = sqrt(self.metrics['mse']) / epochs
        cdls = research(cheese_fresh[-batch:], study=False)
        cdl_open = float(cdls[0])
        cdl_close = float(cdls[-1])
        cdl_pad = (cdl_open - cdl_close) / (batch - 2)
        predictions.append(cdl_open)
        predictions += [
            cdl_open + (cdl_pad * (n + 1)) for n in batch_range
            ]
        predictions.append(cdl_close)
        self.metrics['predictions'] = predictions
        self.__time_plot__(self.metrics['predictions'], cheese_aged)
        self.__manage_state__(call_type=1)
        return True
