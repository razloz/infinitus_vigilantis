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
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ConstantLR
from torch.optim.lr_scheduler import LinearLR
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ChainedScheduler
from torch.optim.lr_scheduler import SequentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
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
            'lr': 0.3,
            'momentum': 0.9,
            'weight_decay': 0.3,
            'nesterov': True,
            'maximize': False,
            'foreach': True
            }
        # Atropos the mouse, sister of Clotho and Lachesis.
        Atropos = self._Atropos_ = ParameterDict()
        Atropos['name'] = 'Atropos'
        # Clotho the mouse, sister of Lachesis and Atropos.
        Clotho = self._Clotho_ = ParameterDict()
        Clotho['name'] = 'Clotho'
        # Lachesis the mouse, sister of Atropos and Clotho.
        Lachesis = self._Lachesis_ = ParameterDict()
        Lachesis['name'] = 'Lachesis'
        for mouse in [Atropos, Clotho, Lachesis]:
            mouse['candles'] = None
            mouse['loss_fn'] = MSELoss()
            mouse['lstm'] = LSTM(**lstm_params)
            mouse['metrics'] = {}
            mouse['optim'] = SGD(mouse['lstm'].parameters(), **opt_params)
            mouse['schedulers'] = [
                ExponentialLR(mouse['optim'], 0.003),
                LinearLR(mouse['optim'], total_iters=89),
                ]
            mouse['warm_lr'] = CosineAnnealingWarmRestarts(mouse['optim'], 89)
            mouse['chainedscheduler'] = ChainedScheduler(mouse['schedulers'])
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

    def __time_step__(self, mouse, inputs, targets, study=False, ts=0):
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
            mouse['warm_lr'].step()
            diff = mouse['candles'] - targets
            mouse['metrics']['mae'] += torch.abs(diff).sum().data
            mouse['metrics']['mse'] += (diff ** 2).sum().data
        else:
            with torch.no_grad():
                try:
                    mouse['metrics']['mae'] = mouse['metrics']['mae'] / ts
                    mouse['metrics']['mse'] = sqrt(mouse['metrics']['mse'] / ts)
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
        moirai = [Atropos, Clotho, Lachesis]
        target_loss = 0.618
        timeout = 89
        epochs = 0
        final_loss = 1
        while final_loss > target_loss:
            if epochs == timeout: break
            inputs = [[],[],[]]
            targets = [[],[],[]]
            batch_count = 0
            final_loss = 1
            for mouse in moirai:
                mouse['metrics']['mae'] = 0
                mouse['metrics']['mse'] = 0
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
                time_step(
                    mouse,
                    mouse['data_final'].clone(),
                    None,
                    ts=timestamps
                    )
                name = mouse['name']
                if name in ['Clotho', 'Atropos']:
                    final_loss += mouse['metrics']['mae']
                mouse['chainedscheduler'].step()
                if self.verbose:
                    print(prefix,'LR:',mouse['chainedscheduler'].get_last_lr())
                    for k in mouse['metrics'].keys():
                        print(f'{prefix} {k}: {mouse["metrics"][k]}')
            epochs += 1
            final_loss = float(final_loss.mean())
            if final_loss != 0:
                final_loss = (final_loss / epochs) ** 2
            if self.verbose:
                msg = '({}) A moment of research yielded a final loss of {}'
                if self.verbose:
                    print(prefix, msg.format(epochs, final_loss))
        sealed = list()
        for mouse in moirai:
            sealed.append(mouse['candles'].detach().cpu().numpy())
            if self.verbose:
                print(mouse['name'], mouse['metrics']['mae'])
        last_pred = float(sealed[0][-1].item())
        last_price = float(Atropos['data_final'][-1:, 3:4].item())
        proj_gain = percent_change(last_pred, last_price)
        self.predictions[symbol] = {
            'final_loss': final_loss,
            'sealed_candles': sealed,
            'last_price': last_price,
            'num_epochs': epochs,
            'metrics': dict(mouse['metrics']),
            'proj_gain': proj_gain,
            'proj_time': time.time()
            }
        if self.verbose:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, a loss of {} was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_loss))
        data_labels = ['data_inputs', 'data_targets', 'data_final']
        for label in data_labels:
            Clotho[label] = None
            Lachesis[label] = None
            Atropos[label] = None
        self.__manage_state__(call_type=1)
        return True

