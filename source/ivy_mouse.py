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
        self.offerings = offerings.clone()
        constants = {}
        constants['cook_time'] = cook_time
        constants['dropout'] = iota
        constants['eps'] = iota * 1e-6
        constants['n_syms'] = len(self.symbols)
        constants['hidden'] = constants['n_syms']
        constants['layers'] = 2
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
        self.cauldron = nn.GRU(
            input_size=constants['n_syms'],
            hidden_size=constants['hidden'],
            num_layers=constants['layers'],
            bias=True,
            batch_first=False,
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
            [int(constants['n_syms'])],
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
        self.candelabrum = self.candelabrum.softmax(0).requires_grad_(True)
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

    def __time_step__(self, candle, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        if study:
            self.train()
            self.optimizer.zero_grad()
            return self.forward(candle)
        else:
            self.eval()
            with torch.no_grad():
                return self.forward(candle)

    def forward(self):
        """**bubble*bubble**bubble**"""
        candles = self.candelabrum
        bubbles = candles.view(1, candles.shape[0])
        bubbles = self.cauldron(bubbles)[1].softmax(0)
        print(bubbles.shape)
        sigil = torch.topk(bubbles, 13, dim=0, largest=True)
        return sigil

    def create_sigil(dataframe, sigil_type='weather'):
        """Translate dataframe into an arcane sigil."""
        if sigil_type == 'weather':
            self.summon_storm(dataframe)

    def research(self, offering, tensor_candles):
        """Moirai research session, fully stocked with cheese and drinks."""
        verbosity = self.verbosity
        constants = self._constants_
        prefix = self._prefix_
        cook_time = constants['cook_time']
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        research = self.__time_step__
        cooking = True
        t_cook = time.time()
        while cooking:
            compass = list()
            trades = list()
            signals = list()
            trading = False
            entry = float(0)
            trade_days = 0
            trade_count = 0
            trade_span = 0
            avg_span = 0
            avg_gain = 0
            net_gains = 0
            correct = 0
            accuracy = 0
            for day in candle_range:
                add_signal = False
                candle = fresh_cheese[day]
                sigil = research(candle, study=True)
                sentiment = sigil.values.item()
                signal = sigil.indices.item()
                compass.append(sentiment)
                trade = no_trade.clone()
                if trading:
                    trade_days += 1
                    if signal == 2 and trade_days >= 3:
                        trading = False
                        add_signal = True
                        day_avg = candle[-1]
                        trade = (day_avg - entry) / entry
                        if trade > 0:
                            correct += 1
                        entry = float(0)
                        net_gains += trade.item()
                        trade_span += trade_days
                        trade_count += 1
                        accuracy = correct / trade_count
                        avg_span = trade_span / trade_count
                        avg_gain = net_gains / trade_count
                        if verbosity > 2:
                            details = msg.format(day_avg, sentiment)
                            print(prefix, 'exited', details)
                            print(prefix, 'trade_days', trade_days)
                            print(prefix, 'trade', trade.item())
                            print(prefix, 'net_gains', net_gains)
                            print(prefix, 'trade_count', trade_count)
                else:
                    if signal == 0:
                        trading = True
                        add_signal = True
                        trade_days = 0
                        entry = float(candle[-1].item())
                        if verbosity > 2:
                            details = msg.format(entry, sentiment)
                            print('')
                            print(prefix, 'entered', details)
                loss = loss_fn(trade, max_trade)
                loss.backward()
                optimizer.step()
                trades.append(trade.item())
                if add_signal:
                    signals.append(signal)
                else:
                    signals.append(0)
            self.epochs += 1
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
                sigil_path = f'{self._sigil_path_}/{symbol}.sigil'
                with open(sigil_path, 'w+') as file_obj:
                    file_obj.write(json.dumps({
                        'symbol': symbol,
                        'signal': signals[-1],
                        'sentiment': sentiment,
                        'net_gains': net_gains,
                        'trade_count': trade_count,
                        'trading': trading,
                        'entry': entry,
                        'trade_days': trade_days,
                        'avg_span': avg_span,
                        'avg_gain': avg_gain,
                        'accuracy': accuracy,
                        'compass': compass,
                        'trades': trades,
                        'signals': signals,
                        }))
                if verbosity > 1:
                    print('***************************************************')
                    print(prefix, 'symbol', symbol)
                    print(prefix, 'signal', signals[-1])
                    print(prefix, 'sentiment', sentiment)
                    print(prefix, 'net_gains', net_gains)
                    print(prefix, 'trade_count', trade_count)
                    print(prefix, 'trading', trading)
                    print(prefix, 'entry', entry)
                    print(prefix, 'trade_days', trade_days)
                    print(prefix, 'avg_span', avg_span)
                    print(prefix, 'avg_gain', avg_gain)
                    print(prefix, 'accuracy', accuracy)
                    print('***************************************************')
        self.__manage_state__(call_type=1)
        return True

    def summon_storm(sigil):
        """The clouds gather to the sounds of song."""
        pass
