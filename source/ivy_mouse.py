"""Three blind mice to predict the future."""
import json
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
from os import listdir, mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbols, offerings, cook_time=0, verbosity=2):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__()
        iota = 1 / 137
        phi = 0.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._sigil_path_ = abspath('./rnn/sigil')
        if not exists(self._sigil_path_): mkdir(self._sigil_path_)
        self.symbols = list(symbols)
        self.offerings = offerings.clone().detach()
        constants = {}
        constants['cook_time'] = cook_time
        constants['dropout'] = iota
        constants['eps'] = iota * 1e-6
        constants['n_syms'] = len(self.symbols)
        constants['input_size'] = constants['n_syms']
        constants['hidden'] = constants['n_syms']
        constants['layer'] = 32
        constants['proj_size'] = constants['n_syms']
        constants['prob_init'] = 1 / constants['n_syms']
        constants['lr_decay'] = iota / phi
        constants['lr_init'] = iota ** phi
        constants['mask_prob'] = (phi - 1)
        constants['momentum'] = phi * phi
        constants['tolerance'] = (iota * phi) / 3
        constants['weight_decay'] = iota / (1 + phi)
        self._constants_ = dict(constants)
        self._prefix_ = prefix = 'Moirai:'
        self._symbol_ = None
        self.cauldron = nn.LSTMCell(
            input_size=constants['input_size'],
            hidden_size=constants['hidden'],
            bias=True,
            device=self._device_,
            dtype=torch.float,
        )
        self.wax_layer = torch.nn.Bilinear(
            in1_features=constants['n_syms'],
            in2_features=constants['n_syms'],
            out_features=constants['n_syms'] ** 3,
            bias=True,
            device=self._device_,
            dtype=torch.float,
            )
        self.wax_coat = torch.nn.Conv3d(
            in_channels=constants['n_syms'] ** 3,
            out_channels=constants['n_syms'],
            kernel_size=9,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros',
            device=self._device_,
            dtype=torch.float,
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
        self.mask_p = lambda t: torch.full_like(t, phi, device=self._device_)
        self.mask = lambda t: t * torch.bernoulli(self.mask_p(t))
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

    def __sealed_candles__(self, candles, study=False):
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
        bubbles = self.mask(candles)
        self.cauldron(
        return self.cauldron(
            self.candelabrum.unsqueeze(0),
            (self.wax_hidden, self.wax_cell),
            )[0].log_softmax(1)

    def create_sigil(dataframe, sigil_type='weather'):
        """Translate dataframe into an arcane sigil."""
        if sigil_type == 'weather':
            self.summon_storm(dataframe)

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        banner = ''.join(['*' for _ in range(80)])
        constants = self._constants_
        cook_time = constants['cook_time']
        loss_fn = self.loss_fn
        offerings = self.offerings
        optimizer = self.optimizer
        prefix = self._prefix_
        sealed_candles = self.__sealed_candles__
        symbols = self.symbols
        pt_stack = torch.stack
        pt_tensor = torch.tensor
        verbosity = self.verbosity
        symbol_range = range(offerings.shape[0])
        time_range = range(offerings.shape[1])
        cooking = True
        t_cook = time.time()
        change_stack = offerings[: , :, -1].H * 0.01
        losses = 0
        loss_retry = 0
        loss_timeout = 50
        prev_grad = None
        while cooking:
            loss_avg = 0
            for day in time_range:
                changes = change_stack[day, :]
                candles = sealed_candles(changes, study=True)
                changes = changes.log_softmax(0).unsqueeze(0)
                loss = loss_fn(candles, changes)
                print('candles', candles)
                print(candles.shape)
                print('changes', changes)
                print(changes.shape)
                if self.candelabrum.grad is not None:
                    if self.candelabrum.grad is prev_grad:
                        print('no grad change!')
                prev_grad = self.candelabrum.grad
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
            loss_avg = loss_avg / (time_range[-1] + 1)
            self.epochs += 1
            if loss_avg != losses:
                loss_retry = 0
                losses = loss_avg
                print(banner)
                for i in symbol_range:
                    print(prefix, symbols[i], candles[i])
                print(prefix, 'loss', loss_avg)
                print(banner)
            else:
                loss_retry += 1
                if loss_retry == loss_timeout:
                    print(prefix, 'losses', losses)
                    cooking = False
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
        self.__manage_state__(call_type=1)
        return True

    def summon_storm(sigil):
        """The clouds gather to the sounds of song."""
        pass





import torch
dev_type = 'cuda:0' if torch.cuda.is_available() else 'cpu'
dev = torch.device(dev_type)
phi = 0.618033988749894
tfloat = torch.float
mask_p = lambda t: torch.full_like(t, phi, device=dev, dtype=tfloat)
mask = lambda t: t * torch.bernoulli(mask_p(t))
fft = torch.fft
inscription = fft.rfft
wax_layer = torch.nn.Bilinear(
    in1_features=36,
    in2_features=36,
    out_features=9,
    bias=True,
    device=dev,
    dtype=tfloat,
    )
cauldron = torch.nn.LSTMCell(
    input_size=171,
    hidden_size=2048,
    bias=True,
    device=dev,
    dtype=tfloat,
)
wax_coat = torch.nn.LSTMCell(
    input_size=2048,
    hidden_size=2048,
    bias=True,
    device=dev,
    dtype=tfloat,
)
x = torch.randn(171, 36, device=dev, dtype=tfloat).log_softmax(1)
wax = wax_layer(x, x).log_softmax(1)
sigil = inscription(wax)
sigil = torch.stack([sigil.real, sigil.imag])
sigil = ((1 / sigil).mean(2) ** -1).mean(0).sin()
sigil = mask(sigil).log_softmax(0)
candles = cauldron(sigil)[0]
candles = wax_coat(mask(candles))[0]
