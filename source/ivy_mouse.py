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


def read_sigil(num):
    """Translate the inscribed sigils."""
    from pandas import DataFrame
    inscriptions = dict()
    sigils = list()
    sigil_path = abspath('./rnn/sigil')
    keys = ['signal', 'accuracy', 'avg_gain', 'net_gains', 'sentiment']
    ascending = [True, False, False, False, False]
    for file_name in listdir(sigil_path):
        if file_name[-6:] == '.sigil':
            file_path = abspath(f'{sigil_path}/{file_name}')
            with open(file_path, 'r') as file_obj:
                sigil_data = json.loads(file_obj.read())
            symbol = sigil_data['symbol']
            inscriptions[symbol] = sigil_data
            sigil = {'symbol': symbol}
            for key, value in sigil_data.items():
                if key in keys:
                    sigil[key] = value
            sigils.append(sigil)
    sigils = DataFrame(sigils)
    sigils.set_index('symbol', inplace=True)
    sigils.sort_values(keys, ascending=ascending, inplace=True)
    best = list()
    top_sigils = sigils.index[:num]
    for symbol in top_sigils:
        best.append(inscriptions[symbol])
    print(sigils[:num])
    return best


def boil_wax(num):
    """Full reduction numerology."""
    if type(num) != str:
        num = str(num)
    if '.' in num:
        num = num.replace('.', '')
    while len(num) > 1:
        num = str(sum([int(i) for i in num]))
    return int(num)


def golden_brew():
    """Care for a spot of tea?"""
    from math import pi
    from torch import stack, tensor
    from torch.fft import rfft, irfft, rfftfreq
    def transform(brew):
        t = list()
        for f, b in brew.items():
            input_signal = tensor(b, dtype=torch.float)
            freq_domain = rfft(input_signal)
            harmonic_mean = stack([freq_domain.real, freq_domain.imag])
            harmonic_mean = (1 / harmonic_mean).mean(0) ** -1
            if harmonic_mean.shape[0] > 25:
                harmonic_mean = harmonic_mean[:25]
            t.append(harmonic_mean)
        return t
    fib_seq = icy.FibonacciSequencer()
    fib_seq.skip(100000)
    f1, f2 = fib_seq.next(2)
    phi = f2 / f1
    golden_string = f'{(f2 / f1):.50}'.replace('.', '')
    pi_string = f'{pi:.50}'.replace('.', '')
    golden_r = [int(n) for n in golden_string]
    pi_r = [int(n) for n in pi_string]
    golden_brew = list()
    pi_brew = list()
    l = 0
    for i in range(50):
        g = golden_string[:i+1]
        p = pi_string[:i+1]
        m = len(g)
        if m > l:
            l = m
        else:
            break
        golden_brew.append(boil_wax(g))
        pi_brew.append(boil_wax(p))
    t = {'fib': golden_brew, 'pi': pi_brew, 'fib_raw': golden_r, 'pi_raw': pi_r}
    brew = (1 / stack(transform(t), dim=0)).mean(0) ** -1
    return brew.clone()


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbols, offerings, cook_time=0, verbosity=0):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__()
        iota = 1 / 137
        phi = 1.618033988749894
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
        constants['hidden'] = constants['n_syms']
        constants['layers'] = 256
        constants['proj_size'] = constants['n_syms']
        constants['lr_decay'] = iota / (phi - 1)
        constants['lr_init'] = iota ** (phi - 1)
        constants['mask_prob'] = phi - 1
        constants['momentum'] = phi * (phi - 1)
        constants['tolerance'] = (iota * (phi - 1)) / 3
        constants['weight_decay'] = iota / phi
        self._constants_ = dict(constants)
        self._prefix_ = prefix = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self.cauldron = nn.LSTM(
            input_size=1,
            hidden_size=constants['hidden'],
            num_layers=constants['layers'],
            proj_size=1,
            bias=True,
            batch_first=True,
            dropout=constants['dropout'],
            bidirectional=False,
            **self._p_tensor_,
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
            foreach=True,
            maximize=True,
            )
        self.epochs = 0
        self.metrics = dict()
        self.candelabrum = torch.bernoulli(torch.full(
            [constants['n_syms']],
            constants['mask_prob'],
            **self._p_tensor_,
            ))
        brewing_tea = golden_brew().tolist()
        while len(brewing_tea) <= self.candelabrum.shape[0]:
            brewing_tea += brewing_tea
        self.tea = torch.tensor(
            brewing_tea[:self.candelabrum.shape[0]],
            **self._p_tensor_,
            )
        self.candelabrum *= self.tea
        self.candelabrum = self.candelabrum.softmax(0).unsqueeze(0).H
        self.candelabrum.requires_grad_(True)
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

    def __sealed_candles__(self, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        if study:
            self.train()
            self.optimizer.zero_grad()
            return self.forward()
        else:
            self.eval()
            with torch.no_grad():
                return self.forward()

    def __layer_wax__(self):
        """More bubbles, because...bubbles."""
        constants = self._constants_
        if self.candelabrum.grad is None:
            return False
        grad = self.candelabrum.grad.clone().detach()
        self.candelabrum = torch.bernoulli(torch.full(
            [constants['n_syms']],
            constants['mask_prob'],
            **self._p_tensor_,
            ))
        self.candelabrum *= self.tea
        self.candelabrum = self.candelabrum.softmax(0).unsqueeze(0).H
        self.candelabrum.requires_grad_(True)
        self.candelabrum.grad = grad
        return True

    def forward(self):
        """**bubble*bubble**bubble**"""
        bubbles = self.__layer_wax__()
        return self.cauldron(self.candelabrum)[0].log_softmax(0).squeeze(1)

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
        pt_stack = torch.stack
        pt_tensor = torch.tensor
        verbosity = self.verbosity
        symbol_range = range(offerings.shape[0])
        time_range = range(offerings.shape[1])
        cooking = True
        t_cook = time.time()
        change_stack = (1 - (offerings[: , :, -1].H * 0.01)).tanh().softmax(1)
        losses = 0
        loss_retry = 0
        loss_timeout = 50
        while cooking:
            loss_avg = 0
            for day in time_range:
                candles = sealed_candles(study=True)
                changes = change_stack[day, :]
                loss = loss_fn(candles, changes)
                loss.backward()
                optimizer.step()
                loss_avg += loss.item()
            loss_avg = loss_avg / (time_range[-1] + 1)
            self.epochs += 1
            if loss_avg != losses:
                loss_poll = 0
                losses = loss_avg
                print(banner)
                for i in symbol_range:
                    print(prefix, symbols[i], candles[i])
                print(prefix, 'loss', loss_avg)
                print(banner)
            else:
                loss_poll += 1
                if loss_poll == loss_timeout:
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
