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
from torch import bernoulli, fft, full_like, nn, stack
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt, inf
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
        constants['n_sample'] = n_sample = 34
        constants['n_symbols'] = n_symbols = len(self.symbols)
        constants['n_output'] = n_sample * n_symbols
        constants['n_time'] = int(offerings.shape[0] - n_sample)
        constants['n_lin_in'] = n_lin_in = 18
        constants['n_lin_out'] = n_lin_out = 9
        constants['output_dims'] = [n_sample, n_symbols, n_lin_out]
        self._constants_ = dict(constants)
        self._prefix_ = prefix = 'Moirai:'
        tfloat = torch.float
        self.mask_p = lambda t: full_like(t, phi, device=dev, dtype=tfloat)
        self.mask = lambda t: self.iota + (t * bernoulli(self.mask_p(t)))
        self.rfft = fft.rfft
        self.signals = offerings[:, :, 0].clone().detach()
        self.signals.requires_grad_(True)
        self.targets = offerings[:, :, 1].clone().detach()
        self.candles = offerings[:, :, 2].clone().detach()
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
        self.loss_fn = nn.HuberLoss(
            reduction='mean',
            delta=1.0,
            )
        self.optimizer = RMSprop(
            self.parameters(),
            foreach=True,
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
        bubbles = self.rfft(signal.transpose(0, 1))
        bubbles = self.bilinear(bubbles.real, bubbles.imag)
        bubbles = self.mask(bubbles.transpose(0, 1))
        bubbles = self.mask(self.input_cell(bubbles)[0])
        bubbles = self.output_cell(bubbles)[0]
        bubbles = self.linear(bubbles).view(self._constants_['output_dims'])
        return bubbles.sum(2).softmax(1).clone()

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        banner = ''.join(['*' for _ in range(80)])
        constants = self._constants_
        cook_time = constants['cook_time']
        inscribe_sigil = self.inscribe_sigil
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        prefix = self._prefix_
        symbols = self.symbols
        verbosity = self.verbosity
        n_time = constants['n_time']
        symbol_range = range(constants['n_symbols'])
        last_batch = self.signals[-constants['n_sample']:]
        last_range = range(last_batch.shape[0])
        cooking = True
        t_cook = time.time()
        losses = inf
        loss_retry = 0
        loss_timeout = 13
        self.train()
        while cooking:
            loss_avg = 0
            for signal, target, candle in iter(self.cauldron):
                optimizer.zero_grad()
                sigil = inscribe_sigil(signal)
                print(sigil)
                print('sigil', sigil.shape)
                loss = loss_fn(sigil, target)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
                print(candles)
                print(candles.shape)
                print(loss)
            loss_avg = loss_avg / n_time
            self.epochs += 1
            print(prefix, f'epoch({self.epochs}), loss_avg({loss_avg})')
            if loss_avg < losses:
                loss_retry = 0
                losses = loss_avg
            else:
                loss_retry += 1
                if loss_retry == loss_timeout:
                    cooking = False
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
            self.__manage_state__(call_type=1)
        self.eval()
        sigil = list()
        for day in last_range:
            candles = inscribe_sigil(
                last_batch[day],
                None,
                study=False,
                )
            sigil.append(candles.clone())
        sigil = stack(sigil)
        print(sigil)
        print('sigil', sigil.shape)
        print(banner)
        for i in symbol_range:
            print(prefix, symbols[i], candles[i])
        print(prefix, 'losses', losses)
        print(banner)
        return True
