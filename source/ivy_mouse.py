"""Three blind mice to predict the future."""
import json
import hashlib
import io
import time
import torch
import traceback
import matplotlib.pyplot as plt
import numpy as np
import source.ivy_commons as icy
from torch import bernoulli, fft, full_like, nan_to_num, nn, stack
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset
from math import inf
from os import listdir, mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbols, offerings, cook_time=inf, verbosity=2):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.iota = iota = 1 / 137
        self.phi = phi = 0.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = dev = torch.device(self._device_type_)
        self.to(dev)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._sigil_path_ = abspath('./rnn/sigil')
        if not exists(self._sigil_path_): mkdir(self._sigil_path_)
        self.symbols = list(symbols)
        offerings = offerings.to(dev)
        constants = {}
        constants['cook_time'] = cook_time
        constants['hidden'] = 2048
        constants['n_sample'] = n_sample = 3
        constants['n_symbols'] = n_symbols = len(self.symbols)
        constants['n_output'] = n_sample * n_symbols
        constants['n_time'] = int(offerings.shape[0] - n_sample)
        constants['n_lin_in'] = n_lin_in = 2
        constants['n_lin_out'] = n_lin_out = 9
        constants['max_trades'] = 1
        constants['output_dims'] = [n_sample, n_symbols, n_lin_out]
        while constants['n_time'] % n_sample != 0:
            constants['n_time'] -= 1
        constants['max_gain'] = 137
        offerings = offerings[-constants['n_time']:]
        self._constants_ = dict(constants)
        self._prefix_ = prefix = 'Moirai:'
        tfloat = torch.float
        self.mask_p = lambda t: full_like(t, phi, device=dev, dtype=tfloat)
        self.mask = lambda t: self.iota + (t * bernoulli(self.mask_p(t)))
        self.rfft = fft.rfft
        self.signals = offerings[:, :, 0].clone().detach()
        self.signals.requires_grad_(True)
        self.targets = offerings[:, :, 1].clone().detach()
        self.targets = (self.targets + iota).softmax(1)
        self.target_gain = torch.tensor(
            constants['max_gain'],
            device=dev,
            dtype=tfloat,
            )
        self.target_zero = torch.zeros(1, device=dev, dtype=tfloat).squeeze()
        self.candles = offerings[:, :, 2].clone().detach()
        self.candles.requires_grad_(True)
        self.candelabrum = TensorDataset(
            self.signals[:-n_sample],
            self.targets[n_sample:],
            self.candles[:-n_sample],
            )
        self.cauldron = DataLoader(self.candelabrum, batch_size=n_sample)
        self.bilinear = nn.Bilinear(
            in1_features=n_lin_in,
            in2_features=n_lin_in,
            out_features=n_lin_out,
            bias=True,
            device=dev,
            dtype=tfloat,
            )
        self.input_cell = nn.LSTMCell(
            input_size=constants['n_symbols'],
            hidden_size=constants['hidden'],
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.output_cell = nn.LSTMCell(
            input_size=constants['hidden'],
            hidden_size=constants['hidden'],
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.linear = nn.Linear(
            in_features=constants['hidden'],
            out_features=constants['n_output'],
            bias=True,
            device=dev,
            dtype=tfloat,
            )
        self.normalizer = nn.InstanceNorm1d(
            num_features=constants['n_symbols'],
            eps=1e-09,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
            device=dev,
            dtype=tfloat,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='mean',
            delta=1.0,
            )
        self.optimizer = RMSprop(
            self.parameters(),
            foreach=True,
            maximize=False,
            )
        self.epochs = 0
        self.metrics = dict()
        self.verbosity = int(verbosity)
        if self.verbosity > 1:
            for key, value in constants.items():
                print(prefix, f'set {key} to {value}')
            print('')
        self.__manage_state__(call_type=0)

    def __manage_state__(self, call_type=0, singular=True):
        """Handles loading and saving of the RNN state."""
        state_path = f'{self._state_path_}/'
        if singular:
            state_path += '.norn.state'
        else:
            state_path += f'.{self._symbol_}.state'
        if call_type == 0:
            try:
                state = torch.load(state_path, map_location=self._device_type_)
                self.load_state_dict(state['moirai'])
                self.epochs = state['epochs']
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            except FileNotFoundError:
                if not singular:
                    self.epochs = 0
                if self.verbosity > 2:
                    print(self._prefix_, 'No state found, creating default.')
                self.__manage_state__(call_type=1)
            except Exception as details:
                if self.verbosity > 1:
                    print(self._prefix_, *details.args)
        elif call_type == 1:
            torch.save(
                {
                    'epochs': self.epochs,
                    'moirai': self.state_dict(),
                    },
                state_path,
                )
            if self.verbosity > 2:
                print(self._prefix_, 'Saved RNN state.')

    def inscribe_sigil(self, signal):
        """
            Let Clotho mold the candles
            Let Lachesis measure the candles
            Let Atropos seal the candles
            Let Awen contain the wax
        """
        iota = self.iota
        mask = self.mask
        bubbles = self.normalizer(signal.transpose(0, 1))
        bubbles = self.rfft(bubbles)
        bubbles = bubbles.real * bubbles.imag
        bubbles = nan_to_num(bubbles, nan=iota, posinf=1.0, neginf=iota)
        bubbles = bubbles.sigmoid().softmax(1)
        bubbles = self.bilinear(bubbles, bubbles)
        bubbles = mask(bubbles.transpose(0, 1))
        bubbles = mask(self.input_cell(bubbles)[0])
        bubbles = self.output_cell(bubbles)[0]
        bubbles = self.linear(bubbles).view(self._constants_['output_dims'])
        bubbles = self.normalizer(bubbles.sum(2)).softmax(1)
        return bubbles.clone()

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        prefix = self._prefix_
        print(prefix, 'Research started.')
        banner = ''.join(['*' for _ in range(80)])
        constants = self._constants_
        cook_time = constants['cook_time']
        inscribe_sigil = self.inscribe_sigil
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        symbols = self.symbols
        topk = torch.topk
        verbosity = self.verbosity
        max_gain = constants['max_gain']
        max_trades = constants['max_trades']
        n_sample = constants['n_sample']
        n_time = constants['n_time']
        sample_range = range(n_sample)
        trade_range = range(max_trades)
        target_gain = self.target_gain
        target_zero = self.target_zero
        least_loss = inf
        cooking = True
        t_cook = time.time()
        loss_retry = 0
        loss_timeout = 137
        msg = '{} epoch({}), loss({}), net_gain({})'
        self.train()
        while cooking:
            net_gain = None
            trades = list()
            loss_avg = 0
            for signal, target, candle in iter(self.cauldron):
                optimizer.zero_grad()
                sigil = inscribe_sigil(signal)
                loss = loss_fn(sigil, target)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                for day in sample_range:
                    trade = topk(sigil[day], max_trades).indices
                    day_avg = candle[day]
                    prev_avg = 0
                    if len(trades) > 0:
                        prev_trade = trades[-1]
                        prev_sym = prev_trade[0]
                        prev_avg = prev_trade[1]
                        trade_avg = day_avg[prev_sym]
                        for i in trade_range:
                            gain = 0
                            _day = trade_avg[i]
                            _prev = prev_avg[i]
                            if _day > 0 < _prev:
                                gain = (_day - _prev) / _prev
                                if net_gain is None:
                                    net_gain = gain.clone()
                                else:
                                    net_gain += gain
                    trades.append((trade, day_avg[trade]))
            if net_gain == 0:
                net_gain = -self.iota
            loss = loss_fn(1 / net_gain, target_zero)
            loss.backward()
            optimizer.step()
            self.signals.grad += self.candles.grad
            loss_avg = loss_avg / n_time
            self.epochs += 1
            print(msg.format(prefix, self.epochs, loss_avg, net_gain))
            if loss_avg < least_loss:
                loss_retry = 0
                least_loss = loss_avg
            else:
                loss_retry += 1
                if loss_retry == loss_timeout:
                    cooking = False
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
        self.__manage_state__(call_type=1)
        self.eval()
        sigil = inscribe_sigil(self.signals[-n_sample:])
        print(banner)
        for t in range(sigil.shape[0]):
            inscriptions = topk(sigil[t], max_trades)
            for i in range(max_trades):
                s = symbols[inscriptions.indices[i]]
                v = inscriptions.values[i]
                print(f'{prefix} day({t}) {s} {v} prob')
        print(msg.format(prefix, self.epochs, loss_avg, net_gain))
        print(banner)
        plt.clf()
        plt.plot(self.signals.grad.clone().detach().cpu().numpy())
        plt.savefig(f'./rnn/signals_grad_{self.epochs}.png')
        plt.clf()
        plt.plot(self.candles.grad.clone().detach().cpu().numpy())
        plt.savefig(f'./rnn/candles_grad_{self.epochs}.png')
        plt.close()
        return True
