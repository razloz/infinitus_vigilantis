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
    """
        golden reduced sequence:
            [1, 7, 8, 7, 7, 1, 4, 4, 3, 2, 9, 4, 4, 3, 3, 7, 7, 7, 9, 5, 7, 3,
             1, 4, 3, 2, 9, 1, 2, 2, 2, 8, 8, 5, 5, 1, 5, 3, 5, 3, 4, 2, 4, 3,
             8, 9, 5, 2, 4, 9]
        pi reduced sequence:
            [3, 4, 8, 9, 5, 5, 7, 4, 9, 3, 8, 7, 7, 5, 5, 8, 9, 1, 6, 6, 6, 4,
             4, 1, 4, 8, 5, 4, 9, 4, 8, 9, 8, 4, 5, 2, 3, 8, 8, 8, 4, 2, 8, 9,
             7, 8, 7, 5, 1, 1]
        tensor([231.0000+0.0000j,  10.6279-9.7771j,   1.7118+10.7979j,
                 28.5454+3.8341j, -15.4845-5.3200j,  -3.1180-6.0696j,
                 11.3475-18.2860j, -29.7906-20.3471j,  14.2649-11.2684j,
                  2.3197+17.5811j,   3.8262-4.8940j,  -1.6922+9.8052j,
                -15.8754+17.2150j,  -6.5192+18.3398j,  -6.6085+8.6744j,
                 -0.8820+5.0656j,  -9.6344-12.4526j,  -6.4756+1.0582j,
                -21.8599-15.8723j,   0.2556+26.3439j, -11.8262-6.9677j,
                 -4.7404+0.4056j, -18.8316+9.6040j, -12.5305+5.1312j,
                 -9.0300-0.6422j,  23.0000+0.0000j])
        tensor([283.0000+0.0000j,   1.8187+1.1361j, -16.1084-8.6592j,
                -12.2353+14.1527j, -18.6677-21.7587j,  -3.7188-7.0207j,
                -18.8479-13.2560j, -14.0505-26.7598j,  15.5139-19.2769j,
                 -8.5556+15.5597j,  -3.0172-6.2084j, -21.4697-16.1817j,
                 -0.4028+4.9231j,  15.7974+1.4659j,   8.0039+4.9104j,
                -13.7812+5.6533j,  16.1623+14.7738j, -16.9695-2.3784j,
                 11.4004+1.2822j,  15.9989+2.4623j,  11.5172-9.0943j,
                -13.7471+0.2094j,  -2.6776-3.6254j,  -4.0873+8.4089j,
                 -6.8760+11.7945j,  25.0000+0.0000j])
    """
    from math import pi
    def transform(brew):
        from numpy import real
        from numpy import imag
        from matplotlib import pyplot
        from torch.fft import rfft
        tensor = torch.tensor
        t = list()
        pyplot.clf()
        for f, b in brew.items():
            ft = rfft(tensor(b))
            pyplot.plot((real(ft) + imag(ft)).sin(), label=f)
            t.append(ft)
        pyplot.savefig(f'./transformed_brew.png')
        pyplot.close()
        return t
    print('pi ratio:', pi)
    print('pi reduction:', boil_wax(pi))
    fib_seq = icy.FibonacciSequencer()
    fib_seq.skip(100000)
    f1, f2 = fib_seq.next(2)
    phi = f2 / f1
    print('golden ratio:', phi)
    print('golden reduction:', boil_wax(phi))
    deci = 100
    golden_string = f'{(f2 / f1):.{deci}}'.replace('.', '')
    pi_string = f'{pi:.{deci}}'.replace('.', '')
    print('golden_string:', golden_string)
    print('pi_string:', pi_string)
    golden_brew = list()
    pi_brew = list()
    l = 0
    for i in range(deci):
        g = golden_string[:i+1]
        p = pi_string[:i+1]
        m = len(g)
        if m > l:
            l = m
        else:
            break
        golden_brew.append(boil_wax(g))
        pi_brew.append(boil_wax(p))
    print('sequence length:', len(golden_brew))
    print('golden reduced sequence:', golden_brew)
    print('pi reduced sequence:', pi_brew)
    return transform({'fib': golden_brew, 'pi': pi_brew})


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, *args, cook_time=0, features=32, verbosity=0, **kwargs):
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
        self._sigil_path_ = abspath('./rnn/sigil')
        if not exists(self._sigil_path_): mkdir(self._sigil_path_)
        self._constants_ = constants = {
            'cook_time': cook_time,
            'dropout': iota,
            'eps': iota * 1e-6,
            'features': int(features),
            'heads': 3,
            'hidden': 9 ** 3,
            'layers': 9,
            'lr_decay': iota / (phi - 1),
            'lr_init': iota ** (phi - 1),
            'mask_prob': phi - 1,
            'momentum': phi * (phi - 1),
            'tolerance': (iota * (phi - 1)) / 3,
            'trine': int((9 ** 3) / 3),
            'weight_decay': iota / phi,
            }
        self._prefix_ = prefix = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self.cauldron = nn.GRU(
            input_size=,
            hidden_size=constants['hidden'],
            num_layers=constants['layers'],
            bias=True,
            batch_first=True,
            dropout=constants['dropout'],
            bidirectional=False,
            **self._p_tensor_,
        )
        self.wax = nn.Bilinear(
            in1_features=constants['features'],
            in2_features=constants['features'],
            out_features=constants['hidden'],
            bias=False,
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

    def forward(self, candle):
        """**bubble*bubble**bubble**"""
        candle = self.wax(candle, candle)

        bubbles = torch.full_like(candle, self._constants_['mask_prob'])
        bubbles = (candle * torch.bernoulli(bubbles)).unsqueeze(0)
        bubbles = self.cauldron(bubbles, bubbles)
        sigil = bubbles.view(3, self._constants_['trine']).sum(1)
        return torch.topk(sigil, 1, dim=0, largest=True, sorted=False)

    def create_sigil(dataframe, sigil_type='weather'):
        """Translate dataframe into an arcane sigil."""
        if sigil_type == 'weather':
            self.summon_storm(dataframe)

    def research(self, offering, tensor_candles):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        if symbol not in self.paterae.keys():
            paterae[symbol] = torch.bernoulli(torch.full(0.34))
            paterae[symbol].requires_grad_(True)
        verbosity = self.verbosity
        constants = self._constants_
        epsilon = constants['eps']
        data_len = len(dataframe)
        prefix = self._prefix_
        cook_time = constants['cook_time']
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        research = self.__time_step__
        p_tensor = self._p_tensor_
        tensor = torch.tensor
        fresh_cheese = tensor(
            dataframe.to_numpy(),
            requires_grad=True,
            **p_tensor,
            )
        no_trade = tensor(float(0), requires_grad=True, **p_tensor)
        max_trade = tensor(float('inf'), **p_tensor)
        candle_range = range(fresh_cheese.shape[0])
        msg = 'trade at {} with a sentiment of {}'
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
