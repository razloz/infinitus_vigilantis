"""Three blind mice to predict the future."""
import time
import torch
import traceback
import source.ivy_commons as icy
from math import sqrt
from os.path import abspath
from torch.utils.data import default_collate
from torch.nn import MSELoss, LSTM, Module, Parameter, ParameterDict
from torch.optim import SGD
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, n_features, n_targets, verbose=False, *args, **kwargs):
        """Inputs: n_features and n_targets must be of type int()"""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._state_path_ = abspath('./rnn/moirai.state')
        self._batch_size_ = batch_size = 13
        self._n_features_ = n_features = int(n_features)
        self._n_targets_ = n_targets = int(n_targets)
        self._n_hidden_ = n_hidden = int(n_features ** 2)
        self._tensor_args_ = {
            'device': self._device_,
            'dtype': torch.float,
            'requires_grad': True
            }
        lstm_params = {
            'input_size': n_features,
            'hidden_size': n_hidden,
            'proj_size': 1,
            'batch_first': True,
            'num_layers': batch_size,
            'dropout': 0.34,
            'device': self._device_
            }
        opt_params = {
            'lr': 0.03,
            'momentum': 0.99,
            'weight_decay': 0.99,
            'nesterov': True,
            'maximize': True,
            'foreach': False
            }
        # Clotho the mouse, sister of Lachesis and Atropos.
        Clotho = self._Clotho_ = ParameterDict()
        Clotho['candles'] = None
        Clotho['data_final'] = None
        Clotho['data_inputs'] = None
        Clotho['data_targets'] = None
        Clotho['loss_fn'] = MSELoss()
        Clotho['lstm'] = LSTM(**lstm_params)
        Clotho['metrics'] = {'accuracy': 0, 'mse': 0, 'mae': 0}
        Clotho['name'] = 'Clotho'
        Clotho['optim'] = SGD(Clotho['lstm'].parameters(), **opt_params)
        # Lachesis the mouse, sister of Atropos and Clotho.
        Lachesis = self._Lachesis_ = ParameterDict()
        Lachesis['candles'] = None
        Lachesis['data_final'] = None
        Lachesis['data_inputs'] = None
        Lachesis['data_targets'] = None
        Lachesis['loss_fn'] = MSELoss()
        Lachesis['lstm'] = LSTM(**lstm_params)
        Lachesis['metrics'] = {'accuracy': 0, 'mse': 0, 'mae': 0}
        Lachesis['name'] = 'Lachesis'
        Lachesis['optim'] = SGD(Lachesis['lstm'].parameters(), **opt_params)
        # Atropos the mouse, sister of Clotho and Lachesis.
        Atropos = self._Atropos_ = ParameterDict()
        Atropos['candles'] = None
        Atropos['data_final'] = None
        Atropos['data_inputs'] = None
        Atropos['data_targets'] = None
        Atropos['loss_fn'] = MSELoss()
        Atropos['lstm'] = LSTM(**lstm_params)
        Atropos['metrics'] = {'accuracy': 0, 'mse': 0, 'mae': 0}
        Atropos['name'] = 'Atropos'
        Atropos['optim'] = SGD(Atropos['lstm'].parameters(), **opt_params)
        self.predictions = ParameterDict()
        self.to(self._device_)
        self.verbose = verbose
        if self.verbose:
            print(self._prefix_, f'Set device to {self._device_type_.upper()}.')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            p = self._state_path_
            if call_type == 0:
                d = self._device_type_
                self.load_state_dict(torch.load(p, map_location=d))
                if self.verbose:
                    print(self._prefix_, 'Loaded RNN state.')
            elif call_type == 1:
                torch.save(self.state_dict(), p)
                if self.verbose:
                    print(self._prefix_, 'Saved RNN state.')
        except Exception as details:
            if self.verbose:
                print(self._prefix_, 'Encountered an exception.')
                traceback.print_exception(details)

    def __time_step__(self, mouse, inputs, targets, study=False, timestamps=0):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        if study is True:
            inputs = torch.vstack(inputs)
            targets = torch.vstack(targets)
            mouse['optim'].zero_grad()
            mouse['candles'] = mouse['lstm'](inputs)[0]
            mouse['loss_fn'](mouse['candles'], targets).backward()
            mouse['optim'].step()
            diff = mouse['candles'] - targets
            mouse['metrics']['mae'] = float(
                mouse['mae'] + torch.abs(diff).sum().data
                )
            mouse['metrics']['mse'] = float(
                mouse['mse'] + (diff ** 2).sum().data
                )
            mouse['metrics']['accuracy'] = float(
                (diff ** 2) < 0.01).float().mean()
                )
        else:
            with torch.no_grad():
                try:
                    mouse['metrics']['mse'] = sqrt(mouse['mse'] / timestamps)
                    mouse['metrics']['mae'] = mouse['mae'] / timestamps
                except Exception as details:
                    if self.verbose:
                        msg = '{} Encountered an exception.'
                        print(self._prefix_, msg.format(mouse['name']))
                        traceback.print_exception(details)
                finally:
                    mouse['candles'] = mouse['lstm'](inputs)[0]

    def research(self, symbol, candles, min_size=90):
        """Moirai research session, fully stocked with cheese and drinks."""
        if not all((
            len(candles.keys()) == self._n_features_,
            len(candles.index) >= min_size
            )): return False
        self.__manage_state__(call_type=0)
        Clotho = self._Clotho_
        Lachesis = self._Lachesis_
        Atropos = self._Atropos_
        time_step = self.__time_step__
        batch = self._batch_size_
        prefix = self._prefix_
        t_args = self._tensor_args_
        percent_change = icy.percent_change
        tensor = torch.tensor
        c_targets = candles['open'].to_numpy()
        l_targets = candles['volume'].to_numpy()
        a_targets = candles['close'].to_numpy()
        timestamps = len(candles.index)
        candles_index = range(timestamps - batch)
        candles = candles.to_numpy()
        t_args['requires_grad'] = True
        Clotho['data_inputs'] = tensor(candles[:-batch], **t_args)
        Lachesis['data_inputs'] = tensor(candles[:-batch], **t_args)
        Atropos['data_inputs'] = tensor(candles[:-batch], **t_args)
        t_args['requires_grad'] = False
        Clotho['data_final'] = tensor(candles[-batch:], **t_args)
        Lachesis['data_final'] = tensor(candles[-batch:], **t_args)
        Atropos['data_final'] = tensor(candles[-batch:], **t_args)
        Clotho['data_targets'] = tensor(c_targets[batch:], **t_args)
        Lachesis['data_targets'] = tensor(l_targets[batch:], **t_args)
        Atropos['data_targets'] = tensor(a_targets[batch:], **t_args)
        moirai = [Clotho, Lachesis, Atropos]
        target_accuracy = 99.95
        timeout = 3
        epochs = 0
        final_accuracy = 0
        while final_accuracy < target_accuracy:
            if epochs == timeout: break
            inputs = [[],[],[]]
            targets = [[],[],[]]
            batch_count = 0
            final_accuracy = 0
            for mouse in moirai:
                mouse['mse'] = Parameter(0.0)
                mouse['mae'] = Parameter(0.0)
            for i in candles_index:
                if batch_count == batch:
                    for m_i, mouse in enumerate(moirai):
                        time_step(mouse, inputs[m_i], targets[m_i], study=True)
                    inputs = [[],[],[]]
                    targets = [[],[],[]]
                    batch_count = 0
                else:
                    for m_i, mouse in enumerate(moirai):
                        inputs[m_i].append(mouse['data_inputs'][i].clone())
                        targets[m_i].append(mouse['data_targets'][i].clone())
                    batch_count += 1
            for mouse in moirai:
                print(mouse['name'], 'MSError', mouse['mse'])
                print(mouse['name'], 'MAError', mouse['mae'])
                time_step(
                    mouse,
                    mouse['data_final'].clone(),
                    None,
                    timestamps=timestamps
                    )
            final_accuracy += float(Clotho['accuracy'][-1])
            final_accuracy += float(Atropos['accuracy'][-1])
            final_accuracy = round(final_accuracy * 0.5, 2)
            epochs += 1
            if self.verbose:
                msg = '({}) A moment of research yielded an accuracy of {}%'
                print(prefix, msg.format(epochs, final_accuracy))
        sealed = list()
        for mouse in moirai:
            sealed.append(mouse['candles'].detach().cpu().numpy())
            print(mouse['name'], mouse['accuracy'])
        last_pred = float(sealed[1][-1])
        last_price = float(Atropos['data_final'][-1][-1].item())
        proj_gain = percent_change(last_pred, last_price)
        self.predictions[symbol] = {
            'sealed_candles': sealed,
            'last_price': last_price,
            'num_epochs': epochs,
            'proj_accuracy': final_accuracy,
            'proj_gain': proj_gain,
            'proj_time': time.time()
            }
        if self.verbose:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, a prediction accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
        data_labels = ['data_inputs', 'data_targets', 'data_final']
        for label in data_labels:
            Clotho[label] = None
            Lachesis[label] = None
            Atropos[label] = None
        self.__manage_state__(call_type=1)
        return True

