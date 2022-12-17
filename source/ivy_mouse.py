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
    def __init__(self, *args, cook_time=0, features=28, verbosity=0, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        iota = 1 / 137
        phi = 1.618033988749894
        features = features - 4
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
            'cluster_shape': 3,
            'cook_time': cook_time,
            'dim': 32,
            'dropout': iota,
            'eps': iota * 1e-6,
            'features': int(features),
            'feature_fold': int(features / 2),
            'hidden': 32 ** 2,
            'layers': 6,
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
            nhead=constants['dim'],
            num_encoder_layers=constants['layers'],
            num_decoder_layers=constants['layers'],
            dim_feedforward=constants['hidden'],
            dropout=constants['dropout'],
            activation='gelu',
            layer_norm_eps=constants['eps'],
            batch_first=True,
            norm_first=True,
            **self._p_tensor_,
            )
        self.wax = nn.Bilinear(
            in1_features=constants['feature_fold'],
            in2_features=constants['feature_fold'],
            out_features=constants['hidden'],
            bias=True,
            **self._p_tensor_,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='mean',
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
        self.activation = nn.functional.gelu
        self.epochs = 0
        self.metrics = dict()
        self.verbosity = int(verbosity)
        self.to(self._device_)
        self.__manage_state__(call_type=0)
        if self.verbosity > 1:
            for key, value in constants.items():
                print(prefix, f'set {key} to {value}')
            print('')

    def __chitchat__(self, prediction_tail, target_tail):
        """Chat with the mice about candles and things."""
        prefix = self._prefix_
        for key, value in self.metrics.items():
            if key in ['predictions', 'signals']:
                continue
            elif key == 'acc_pct':
                value = f'{value}%'
            print(prefix, f'{key}:', value)
        print(prefix, 'total epochs:', self.epochs)
        if self.verbosity > 1:
            print(prefix, 'prediction tail:', prediction_tail.tolist())
            print(prefix, 'target tail:', target_tail.tolist())

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
                self.__manage_state__(call_type=1)
            except Exception as details:
                if self.verbosity > 1:
                    print(self._prefix_, *details.args)
        elif call_type == 1:
            torch.save(
                {'epochs': self.epochs, 'moirai': self.state_dict()},
                state_path,
                )
            if self.verbosity > 2:
                print(self._prefix_, 'Saved RNN state.')

    def __time_plot__(self, predictions, targets):
        batch = self._constants_['batch_size']
        metrics = self.metrics
        nans = [[None, None, None, None] for _ in range(batch)]
        fig = plt.figure(figsize=(5.12, 3.84), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.6, 0.6, 0.6))
        ax.set_ylabel('Probability', fontweight='bold')
        ax.set_xlabel('Batch', fontweight='bold')
        y_p = predictions.detach().cpu().numpy()
        y_t = targets.detach().cpu().numpy()
        y_t = np.vstack([y_t, nans])
        ax.plot(y_t, color='#FF9600') #cheddar
        ax.plot(y_p, color='#FFE88E') #gouda
        symbol = self._symbol_
        accuracy = metrics['acc_pct']
        epoch = self.epochs
        signal = metrics['signal']
        title = f'{symbol} ({signal})\n'
        title += f'{accuracy}% correct, '
        title += f'{epoch} epochs.'
        fig.suptitle(title, fontsize=11)
        plt.savefig(f'{self._epochs_path_}/{int(time.time())}.png')
        plt.clf()
        plt.close()

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
        batch = constants['batch_size']
        dim = constants['dim']
        fold = constants['feature_fold']
        inscribe = torch.topk
        mask = constants['mask_prob']
        wax = self.melt(candles)
        wax = self.wax(wax[:, :fold], wax[:, fold:])
        wax = self.activation(wax)
        bubbles = torch.full_like(wax, mask)
        bubbles = torch.bernoulli(bubbles)
        bubbles = wax * bubbles
        bubbles = self.cauldron(bubbles, bubbles)
        bubbles = bubbles.view(bubbles.shape[0], dim, dim)
        candles = inscribe(bubbles, 1)[0].flatten(0)
        candles = inscribe(candles, batch * 4)[0]
        candles = candles.view(batch, 4) ** 2
        return candles.clone()

    def research(self, offering, dataframe, plot=True, keep_predictions=True):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        verbosity = self.verbosity
        constants = self._constants_
        batch = constants['batch_size']
        data_len = len(dataframe)
        batch_error = f'{self._prefix_} {symbol} data less than batch size.'
        if data_len < batch:
            if verbosity > 1:
                print(batch_error)
            return False
        while data_len % batch != 0:
            dataframe = dataframe[1:]
            data_len = len(dataframe)
            if data_len < batch:
                if verbosity > 1:
                    print(batch_error)
                return False
        cook_time = constants['cook_time']
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        p_tensor = self._p_tensor_
        research = self.__time_step__
        tensor = torch.tensor
        vstack = torch.vstack
        self.metrics = {
            'accuracy': [0, 0],
            'acc_pct': 0,
            'loss': 0,
            'predictions': [],
            'signal': 'neutral',
            'signals': [],
            'symbol': symbol,
            'var_delta': 0,
            }
        fresh_cheese = tensor(
            dataframe.to_numpy(),
            requires_grad=True,
            **p_tensor,
            )[:, 4:]
        aged_cheese = tensor(
            dataframe.to_numpy(),
            **p_tensor,
            )[:, :4][batch:]
        sample = TensorDataset(fresh_cheese[:-batch], aged_cheese)
        candles = DataLoader(sample, batch_size=batch)
        cooking = True
        tolerance = aged_cheese.mean(1).mean(0) * constants['tolerance']
        n_targets = batch * 4
        t_cook = time.time()
        while cooking:
            self.metrics['accuracy'] = [0, 0]
            predictions = list()
            signals = list()
            for candle, target in iter(candles):
                coated_candles = research(candle, study=True)
                loss = loss_fn(coated_candles, target)
                loss.backward()
                optimizer.step()
                predictions.append(coated_candles)
                var_coated = coated_candles.var()
                var_target = target.var()
                var_delta = (var_coated - var_target) / var_target
                self.metrics['var_delta'] += var_delta.abs().item()
                adj_loss = sqrt(loss.item()) / 3
                if coated_candles[-1].mean(0) > coated_candles[0].mean(0):
                    signal = 1
                else:
                    signal = 0
                signals.append(signal)
                correct = (coated_candles - target).abs()
                correct = correct[correct <= tolerance].sum()
                self.metrics['accuracy'][0] += correct.item()
                self.metrics['accuracy'][1] += n_targets
                self.metrics['loss'] += adj_loss
            self.epochs += 1
            if time.time() - t_cook >= cook_time:
                cooking = False
        predictions.append(research(fresh_cheese[-batch:], study=False))
        predictions = vstack(predictions)
        correct, epochs = self.metrics['accuracy']
        acc_pct = 100 * (1 + (correct - epochs) / epochs)
        self.metrics['acc_pct'] = round(acc_pct, 2)
        signal = signals[-1]
        if signal == 1:
            self.metrics['signal'] = 'buy'
        else:
            self.metrics['signal'] = 'sell'
        self.metrics['var_delta'] = self.metrics['var_delta'] / epochs
        self.metrics['var_delta'] *= 100
        self.metrics['loss'] = self.metrics['loss'] / epochs
        if keep_predictions:
            sigil_path = f'{self._sigil_path_}/{symbol}.sigil'
            self.metrics['signals'] = signals
            self.metrics['predictions'] = predictions.tolist()
            with open(sigil_path, 'w+') as file_obj:
                file_obj.write(json.dumps(self.metrics))
        if verbosity == 1:
            self.__chitchat__(predictions, aged_cheese)
        if plot:
            self.__time_plot__(predictions, aged_cheese)
        self.__manage_state__(call_type=1)
        return True

    def read_sigil(self, num=0, signal='buy'):
        """Translate the inscribed sigils."""
        from pandas import DataFrame
        ignore = ['accuracy', 'predictions', 'signals']
        inscriptions = dict()
        sigils = list()
        sigil_path = self._sigil_path_
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
            ['signal', 'loss', 'acc_pct', 'var_delta'],
            ascending=[True, True, False, True],
            inplace=True,
            )
        best = list()
        top_sigils = sigils.index[:num]
        for symbol in top_sigils:
            best.append(inscriptions[symbol])
        if self.verbosity > 0:
            print(sigils[:num])
        return best
