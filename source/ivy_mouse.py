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


def read_sigil(num=0, signal='buy'):
    """Translate the inscribed sigils."""
    from pandas import DataFrame
    ignore = ['accuracy', 'predictions', 'signals']
    inscriptions = dict()
    sigils = list()
    sigil_path = abspath('./rnn/sigil')
    for file_name in listdir(sigil_path):
        if file_name[-6:] == '.sigil':
            file_path = abspath(f'{sigil_path}/{file_name}')
            with open(file_path, 'r') as file_obj:
                sigil_data = json.loads(file_obj.read())
            symbol = sigil_data['symbol']
            inscriptions[symbol] = sigil_data
            sigil = dict()
            for key, value in sigil_data.items():
                if key not in ignore:
                    sigil[key] = value
            sigils.append(sigil)
    sigils = DataFrame(sigils)
    sigils.set_index('symbol', inplace=True)
    sigils.sort_values(
        ['signal', 'acc_pct', 'loss', 'var_delta'],
        ascending=[True, False, True, True],
        inplace=True,
        )
    best = list()
    top_sigils = sigils.index[:num]
    for symbol in top_sigils:
        best.append(inscriptions[symbol])
    print(sigils[:num])
    return best


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
            'batch_size': 34,
            'cook_time': cook_time,
            'dim': 9,
            'dropout': iota,
            'eps': iota * 1e-6,
            'features': int(features),
            'heads': 27,
            'hidden': 9 ** 3,
            'layers': 9,
            'lr_decay': iota / (phi - 1),
            'lr_init': iota ** (phi - 1),
            'mask_prob': phi - 1,
            'momentum': phi * (phi - 1),
            'tolerance': (iota * (phi - 1)) / 3,
            'weight_decay': iota / phi,
            }
        self._prefix_ = prefix = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self.cauldron = nn.Transformer(
            d_model=constants['hidden'],
            nhead=constants['heads'],
            num_encoder_layers=constants['layers'],
            num_decoder_layers=constants['layers'],
            dim_feedforward=constants['hidden'],
            dropout=constants['dropout'],
            activation='gelu',
            layer_norm_eps=constants['eps'],
            batch_first=False,
            norm_first=True,
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
        sigil = bubbles.squeeze(0).tanh().mean(0)
        return sigil.clone()

    def create_sigil(dataframe, sigil_type='weather'):
        """Translate dataframe into an arcane sigil."""
        if sigil_type == 'weather':
            self.summon_storm(dataframe)

    def research(self, offering, dataframe):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        verbosity = self.verbosity
        constants = self._constants_
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
            trading = False
            entry = float(0)
            trade_days = 0
            trade_count = 0
            net_gains = 0
            for day in candle_range:
                candle = fresh_cheese[day]
                sentiment = research(candle, study=True).item()
                compass.append(sentiment)
                if verbosity > 3:
                    print(prefix, 'sentiment', sentiment)
                trade = no_trade.clone()
                if trading:
                    trade_days += 1
                    if sentiment < -0.00382:
                        if trading and trade_days > 3:
                            trading = False
                            day_avg = candle[-1]
                            trade = (day_avg - entry) / entry
                            entry = float(0)
                            net_gains += trade.item()
                            trade_count += 1
                            if verbosity > 1:
                                details = msg.format(day_avg, sentiment)
                                print(prefix, 'exited', details)
                                print(prefix, 'trade_days', trade_days)
                                print(prefix, 'trade', trade.item())
                                print(prefix, 'net_gains', net_gains)
                                print(prefix, 'trade_count', trade_count)
                else:
                    if sentiment > 0.00618:
                        trading = True
                        trade_days = 0
                        entry = float(candle[-1].item())
                        if verbosity > 1:
                            details = msg.format(entry, sentiment)
                            print('')
                            print(prefix, 'entered', details)
                loss = loss_fn(trade, max_trade)
                loss.backward()
                optimizer.step()
                trades.append(trade.item())
            self.epochs += 1
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
                sigil_path = f'{self._sigil_path_}/{symbol}.sigil'
                with open(sigil_path, 'w+') as file_obj:
                    file_obj.write(json.dumps({
                        'symbol': symbol,
                        'sentiment': sentiment,
                        'net_gains': net_gains,
                        'trade_count': trade_count,
                        'trading': trading,
                        'entry': entry,
                        'trade_days': trade_days,
                        'compass': compass,
                        'trades': trades,
                        }))
                if verbosity > 1:
                    print('***************************************************')
                    print(prefix, 'symbol', symbol)
                    print(prefix, 'sentiment', sentiment)
                    print(prefix, 'net_gains', net_gains)
                    print(prefix, 'trade_count', trade_count)
                    print(prefix, 'trading', trading)
                    print(prefix, 'entry', entry)
                    print(prefix, 'trade_days', trade_days)
                    print('***************************************************')
        self.__manage_state__(call_type=1)
        return True

    def summon_storm(sigil):
        """The clouds gather to the sounds of song."""
        pass
