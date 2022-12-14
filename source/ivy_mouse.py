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
            'batch_step': 17,
            'cluster_shape': 3,
            'cook_time': cook_time,
            'dim': 5,
            'dropout': iota,
            'eps': iota * 1e-6,
            'features': int(features),
            'hidden': 5 ** 3,
            'layers': 75,
            'lr_decay': iota / (phi - 1),
            'lr_init': iota ** (phi - 1),
            'momentum': phi * (phi - 1),
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
        self.wax = nn.Transformer(
            d_model=constants['layers'] * 27,
            nhead=3,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=constants['hidden'],
            dropout=constants['dropout'],
            activation='gelu',
            batch_first=False,
            **self._p_tensor_,
            )
        self.loss_fn = torch.nn.BCELoss(
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
            #foreach=True,
            maximize=False,
            )
        self.stir = nn.Conv3d(
            in_channels=constants['layers'],
            out_channels=constants['layers'],
            kernel_size=constants['cluster_shape'],
            **self._p_tensor_,
            )
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
            if key == 'predictions':
                continue
            elif key in ['acc_pct', 'confidence', 'avg_conf']:
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
        fig = plt.figure(figsize=(5.12, 3.84), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.6, 0.6, 0.6))
        ax.set_ylabel('Probability', fontweight='bold')
        ax.set_xlabel('Batch', fontweight='bold')
        y_p = predictions.detach().cpu().numpy()
        y_t = targets.detach().cpu().numpy()
        y_t = np.vstack([y_t, [None, None, None]])
        x_range = range(0, y_p.shape[0] * 5, 5)
        n_y = 0
        for r_x in x_range:
            n_x = 0
            for prob in y_t[n_y]:
                if prob is None:
                    continue
                x = r_x + n_x
                ax.plot([x, x], [0, prob], color='#FF9600') #cheddar
                n_x += 1
            n_x = 0
            for prob in y_p[n_y]:
                x = r_x + n_x
                ax.plot([x, x], [0, prob], color='#FFE88E') #gouda
                n_x += 1
            n_y += 1
        symbol = self._symbol_
        accuracy = self.metrics['acc_pct']
        confidence = self.metrics['avg_conf']
        epoch = self.epochs
        signal = self.metrics['signal']
        title = f'{symbol} ({signal})\n'
        title += f'{accuracy}% correct, '
        title += f'{confidence}% confident, '
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
        batch = constants['batch_step']
        dim = constants['dim']
        layers = constants['layers']
        inscribe = torch.topk
        candles = self.melt(candles)
        bubbles = self.cauldron(candles)[1]
        bubbles = bubbles.view(bubbles.shape[0], dim, dim, dim)
        bubbles = self.stir(bubbles).flatten(0).unsqueeze(0)
        wax = self.wax(bubbles, bubbles)
        wax = wax.view(layers, 3, 3, 3)
        candles = inscribe(wax, 1)[0].flatten(0)
        candles = inscribe(candles, 3)[0].softmax(0)
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
        step = constants['batch_step']
        tensor = torch.tensor
        vstack = torch.vstack
        self.metrics = {
            'accuracy': [0, 0],
            'acc_pct': 0,
            'confidence': 0,
            'avg_conf': 0,
            'loss': 0,
            'predictions': [],
            'signal': 'neutral',
            'symbol': symbol,
            }
        cheese = tensor(dataframe.to_numpy(), requires_grad=True, **p_tensor)
        sample = TensorDataset(cheese)
        candles = DataLoader(sample, batch_size=batch)
        target_tmp = tensor([0, 1, 0], **p_tensor)
        cooking = True
        t_cook = time.time()
        while cooking:
            self.metrics['accuracy'] = [0, 0]
            predictions = list()
            targets = list()
            for candle in iter(candles):
                candle = candle[0]
                target = candle[step:]
                candle = candle[:step]
                target_tmp *= 0
                cdl_mean = candle[:, :4].mean(1).mean(0)
                tgt_mean = target[:, :4].mean(1).mean(0)
                if tgt_mean > cdl_mean:
                    target_tmp[0] += 1
                elif tgt_mean < cdl_mean:
                    target_tmp[2] += 1
                else:
                    target_tmp[1] += 1
                coated_candles = research(candle, study=True)
                loss = loss_fn(coated_candles, target_tmp)
                loss.backward()
                optimizer.step()
                predictions.append(coated_candles)
                targets.append(target_tmp.clone())
                adj_loss = sqrt(loss.item()) / 3
                if coated_candles.argmax(0) == target_tmp.argmax(0):
                    correct = 1
                else:
                    correct = 0
                self.metrics['accuracy'][0] += correct
                self.metrics['accuracy'][1] += 1
                self.metrics['loss'] += adj_loss
                if verbosity == 3:
                    self.__chitchat__(predictions[-1], targets[-1])
            self.epochs += 1
            if verbosity == 2:
                self.__chitchat__(predictions[-1], targets[-1])
            if time.time() - t_cook >= cook_time:
                cooking = False
        predictions.append(research(cheese[-step:], study=False))
        predictions = vstack(predictions)
        targets = vstack(targets)
        correct, epochs = self.metrics['accuracy']
        acc_pct = 100 * (1 + (correct - epochs) / epochs)
        self.metrics['acc_pct'] = round(acc_pct, 2)
        signal = predictions[-1].max(0)
        confidence = 100 * signal[0].item()
        self.metrics['confidence'] = round(confidence, 2)
        signal = signal[1].item()
        if signal == 0:
            self.metrics['signal'] = 'buy'
        elif signal == 1:
            self.metrics['signal'] = 'neutral'
        else:
            self.metrics['signal'] = 'sell'
        self.metrics['loss'] = self.metrics['loss'] / epochs
        if keep_predictions:
            sigil_path = f'{self._sigil_path_}/{symbol}.sigil'
            sigils = predictions.max(1)
            inscriptions = targets.max(1)
            nans = [None for _ in range(step)]
            sealed_candles = list()
            for i in range(len(inscriptions.indices)):
                correct = sigils.indices[i] == inscriptions.indices[i]
                sealed_candles += nans
                sealed_candles.append(1 if correct else 0)
                sealed_candles += nans[:-1]
            sealed_candles.append(sigils.values[-1].item())
            self.metrics['avg_conf'] = round(100 * sigils[0].mean(0).item(), 2)
            self.metrics['predictions'] = sealed_candles
            with open(sigil_path, 'w+') as file_obj:
                file_obj.write(json.dumps(self.metrics))
        if verbosity == 1:
            self.__chitchat__(predictions[-2], targets[-1])
        if plot:
            self.__time_plot__(predictions, targets)
        self.__manage_state__(call_type=1)
        return True

    def read_sigil(self, num=0, signal='buy'):
        """Translate the inscribed sigils."""
        sigil_path = self._sigil_path_
        sigil_files = listdir(sigil_path)
        picks = dict()
        for file_name in sigil_files:
            if file_name[-6:] == '.sigil':
                file_path = abspath(f'{sigil_path}/{file_name}')
                with open(file_path, 'r') as file_obj:
                    sigil = json.loads(file_obj.read())
                sym_signal = sigil['signal']
                if sym_signal == signal:
                    avg_conf = sigil['avg_conf']
                    if avg_conf not in picks.keys():
                        picks[avg_conf] = dict()
                    symbol = sigil['symbol']
                    picks[avg_conf][symbol] = sigil
        picks = {k: picks[k] for k in sorted(picks, reverse=True)}
        best = dict()
        for acc_pct, symbols in picks.items():
            for symbol, metrics in symbols.items():
                acc_pct = metrics['acc_pct']
                if acc_pct not in best.keys():
                    best[acc_pct] = dict()
                best[acc_pct][avg_conf] = metrics
        best = {k: best[k] for k in sorted(best, reverse=True)}
        inscribed_candles = list()
        count = 0
        for first_sigil in best.values():
            if count == num:
                break
            for second_sigil in first_sigil.values():
                inscribed_candles.append(second_sigil)
                count = len(inscribed_candles)
                if count == num:
                    break
        if self.verbosity > 0:
            prefix = self._prefix_
            count = 0
            for inscription in inscribed_candles:
                count += 1
                print(prefix, f'#{count} sigil.')
                for key, value in inscription.items():
                    if key in ['predictions', 'signal']:
                        continue
                    elif key in ['acc_pct', 'confidence', 'avg_conf']:
                        value = f'{value}%'
                    print(prefix, f'{key}:', value)
                print('')
        return inscribed_candles
