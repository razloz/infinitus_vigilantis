"""Three blind mice to predict the future."""
import json
import time
import torch
import traceback
import matplotlib.pyplot as plt
import numpy as np
import source.ivy_commons as icy
from torch import bernoulli, fft, full_like, nn, stack
from torch.optim import Adagrad
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
        iota = 1 / 137
        phi = 0.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = dev = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._sigil_path_ = abspath('./rnn/sigil')
        if not exists(self._sigil_path_): mkdir(self._sigil_path_)
        self.symbols = list(symbols)
        self.offerings = offerings.clone().detach().transpose(0, 1)
        constants = {}
        constants['cook_time'] = cook_time
        constants['eps'] = iota * 1e-6
        constants['hidden'] = 2048
        constants['lr_decay'] = iota / phi
        constants['lr_init'] = iota ** phi
        constants['n_features'] = int(self.offerings.shape[2])
        constants['n_sample'] = 34
        constants['n_symbols'] = int(self.offerings.shape[1])
        constants['n_time'] = int(self.offerings.shape[0])
        constants['n_time'] -= constants['n_sample']
        constants['weight_decay'] = iota / (1 + phi)
        self._constants_ = dict(constants)
        self.change_stack = (offerings[: , :, -1] * 0.01).clone().detach().H
        self.change_stack = self.change_stack[constants['n_sample']:].softmax(1)
        self.offerings = self.offerings[:-constants['n_sample']]
        self.offerings.requires_grad_(True)
        self._prefix_ = prefix = 'Moirai:'
        tfloat = torch.float
        self.mask_p = lambda t: full_like(t, phi, device=dev, dtype=tfloat)
        self.mask = lambda t: t * bernoulli(self.mask_p(t))
        self.inscription = fft.rfft
        self.gate = nn.GLU()
        self.leaky = nn.LeakyReLU()
        self.wax_layer = nn.Bilinear(
            in1_features=constants['n_features'],
            in2_features=constants['n_features'],
            out_features=9,
            bias=True,
            device=dev,
            dtype=tfloat,
            )
        self.cauldron = nn.LSTMCell(
            input_size=constants['n_symbols'],
            hidden_size=constants['hidden'],
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.wax_coat = nn.LSTMCell(
            input_size=constants['hidden'],
            hidden_size=constants['hidden'],
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.wax_seal = nn.Linear(
            in_features=int(constants['hidden'] / 2),
            out_features=constants['n_symbols'],
            bias=True,
            device=dev,
            dtype=tfloat,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='mean',
            delta=1.0,
            )
        self.optimizer = Adagrad(
            self.parameters(),
            lr=constants['lr_init'],
            lr_decay=constants['lr_decay'],
            weight_decay=constants['weight_decay'],
            eps=constants['eps'],
            foreach=False,
            maximize=False,
            )
        self.zero_loss = torch.zeros(
            constants['n_symbols'],
            device=dev,
            dtype=tfloat,
            )
        self.epochs = 0
        self.metrics = dict()
        self.verbosity = int(verbosity)
        self.to(self._device_)
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

    def inscribe_sigil(self, candles, changes, study=True):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        if study:
            self.train()
        else:
            self.eval()
        mask = self.mask
        optimizer = self.optimizer
        optimizer.zero_grad()
        bubbles = self.wax_layer(candles, candles)
        bubbles = self.inscription(bubbles)
        bubbles = stack([bubbles.real, bubbles.imag]).mean(2).mean(0)
        bubbles = mask(bubbles).softmax(0)
        bubbles = mask(self.cauldron(bubbles)[0]).softmax(0)
        bubbles = self.wax_coat(bubbles)[0]
        bubbles = self.gate(bubbles).tanh()
        bubbles = self.wax_seal(bubbles).softmax(0)
        if study:
            abs_loss = (bubbles - changes).abs()
            loss = self.loss_fn(abs_loss, self.zero_loss)
            loss.backward()
            optimizer.step()
            return bubbles, loss.item()
        else:
            return bubbles

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        torch.autograd.set_detect_anomaly(True)
        banner = ''.join(['*' for _ in range(80)])
        constants = self._constants_
        cook_time = constants['cook_time']
        inscribe_sigil = self.inscribe_sigil
        offerings = self.offerings
        changes = self.change_stack
        prefix = self._prefix_
        symbols = self.symbols
        verbosity = self.verbosity
        symbol_range = range(constants['n_symbols'])
        time_range = range(constants['n_time'])
        cooking = True
        t_cook = time.time()
        losses = 0
        loss_retry = 0
        loss_timeout = 13
        while cooking:
            loss_avg = 0
            for day in time_range:
                candles, loss = inscribe_sigil(offerings[day], changes[day])
                loss_avg += loss
            loss_avg = loss_avg / constants['n_time']
            self.epochs += 1
            if loss_avg != losses:
                loss_retry = 0
                losses = loss_avg
                print(prefix, 'loss_avg', loss_avg)
            else:
                loss_retry += 1
                if loss_retry == loss_timeout:
                    candles = inscribe_sigil(offerings, None, study=False)
                    print(banner)
                    for i in symbol_range:
                        print(prefix, symbols[i], candles[i])
                    print(prefix, 'losses', losses)
                    print(banner)
                    cooking = False
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
        self.__manage_state__(call_type=1)
        return True
