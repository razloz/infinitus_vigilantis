"""Three blind mice to predict the future."""
import time
import torch
import traceback
import source.ivy_commons as icy
from math import sqrt, log
from os.path import abspath
from torch.nn import BatchNorm1d, Conv3d, GLU, GRU, HuberLoss
from torch.nn import Module, PairwiseDistance, ParameterDict, Sequential
from torch.optim import Rprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as WarmRestarts
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
__split_gru__ = GLU()


class GatedSequence(Sequential):
    """Gated Recurrent Unit output size reduction with abs(tanh) activation."""
    def forward(self, *inputs):
        """For each GRU, splits it, applies activations, and sends to next."""
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(inputs[0])
            else:
                inputs = module(inputs)
            inputs = __split_gru__(inputs[0].tanh().abs())
        return inputs


class ThreeBlindMice(Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, n_features, verbosity=0, min_size=90):
        """Inputs: n_features and n_targets must be of type int()"""
        super(ThreeBlindMice, self).__init__()
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._min_size_ = min_size
        self._prefix_ = 'Moirai:'
        self._state_path_ = abspath('./rnn/moirai.state')
        batch_size = self._batch_size_ = 8
        n_features = self._n_features_ = int(n_features)
        n_gates = 7
        n_hidden = self._batch_size_
        for i in range(n_gates):
            n_hidden *= 2
        print(self._prefix_, 'set batch_size to', batch_size)
        print(self._prefix_, 'set n_features to', n_features)
        print(self._prefix_, 'set n_hidden to', n_hidden)
        print(self._prefix_, 'set n_gates to', n_gates)
        t_args = self._tensor_args_ = dict(
            device=self._device_,
            dtype=torch.float,
            )
        gru_params = dict(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=batch_size,
            bias=True,
            batch_first=True,
            dropout=0.34,
            bidirectional=False,
            )
        opt_params = dict(
            lr=0.01,
            etas=(0.5, 1.2),
            step_sizes=(1e-06, 50),
            foreach=True,
            )
        loss_params = dict(
            reduction='sum',
            delta=0.97,
            )
        # Setup a cauldron for our candles
        cauldron = self._cauldron_ = ParameterDict()
        cauldron['size'] = int(batch_size * 3)
        cauldron['candles'] = None
        cauldron['loss_fn'] = HuberLoss(**loss_params)
        cauldron['loss_targets'] = torch.zeros(batch_size, 1, **t_args)
        cauldron['metrics'] = {}
        gates = list()
        for gate in range(n_gates):
            gates.append(GRU(**gru_params))
            gru_params['hidden_size'] = int(gru_params['hidden_size'] / 2)
            gru_params['input_size'] = int(gru_params['hidden_size'])
        gru_params['hidden_size'] = n_hidden
        gru_params['input_size'] = n_features
        cauldron['nn_gates'] = GatedSequence(*gates)
        cauldron['optim'] = Rprop(cauldron['nn_gates'].parameters(), **opt_params)
        cauldron['warm_lr'] = WarmRestarts(cauldron['optim'], n_hidden)
        print(self._cauldron_)
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
            mouse['loss_fn'] = HuberLoss(**loss_params)
            mouse['loss_targets'] = torch.zeros(batch_size, 1, **t_args)
            mouse['metrics'] = {}
            gates = list()
            for gate in range(n_gates):
                gates.append(GRU(**gru_params))
                gru_params['hidden_size'] = int(gru_params['hidden_size'] / 2)
                gru_params['input_size'] = int(gru_params['hidden_size'])
            gru_params['hidden_size'] = n_hidden
            gru_params['input_size'] = n_features
            mouse['nn_gates'] = GatedSequence(*gates)
            mouse['optim'] = Rprop(mouse['nn_gates'].parameters(), **opt_params)
            mouse['warm_lr'] = WarmRestarts(mouse['optim'], n_hidden)
        self._tensor_args_['requires_grad'] = True
        self.predictions = ParameterDict()
        self.to(self._device_)
        self.verbosity = verbosity
        if self.verbosity > 0:
            print(self._prefix_, f'Set device to {self._device_type_.upper()}.')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            p = self._state_path_
            if call_type == 0:
                d = self._device_type_
                self.load_state_dict(torch.load(p, map_location=d))
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            elif call_type == 1:
                torch.save(self.state_dict(), p)
                if self.verbosity > 2:
                    print(self._prefix_, 'Saved RNN state.')
        except Exception as details:
            if self.verbosity > 2:
                print(self._prefix_, 'Encountered an exception.')
                traceback.print_exception(details)

    def __time_step__(self, mouse, inputs, targets, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        vstack = torch.vstack
        if study is True:
            inputs = vstack(inputs)
            targets = vstack(targets)
            percent_change = icy.percent_change
            batch_size = inputs.shape[0]
            mouse['optim'].zero_grad()
            mouse['candles'] = mouse['nn_gates'](inputs)[0]
            mouse['candles'] = vstack(mouse['candles'].split(1))
            difference = (mouse['candles'] * 3e3 - targets).abs()
            difference = difference.tanh().abs()
            loss = mouse['loss_fn'](difference, mouse['loss_targets'])
            loss.backward()
            mouse['optim'].step()
            mouse['warm_lr'].step()
            mouse['metrics']['mae'] += difference.abs().sum()
            mouse['metrics']['mse'] += (difference ** 2).sum()
            correct = difference[difference == 0].shape[0]
            wrong = batch_size - correct
            confidence = float(1 - loss.item() * 100)
            mouse['metrics']['acc'] = 100 - percent_change(correct, wrong) * -1
            mouse['metrics']['acc'] = round(mouse['metrics']['acc'], 2)
            mouse['metrics']['confidence'] = round(confidence, 2)
        else:
            with torch.no_grad():
                try:
                    for key in ['acc', 'mae', 'mse']:
                        batch_average = mouse['metrics'][key] / epochs
                        if key == 'mse':
                            batch_average = sqrt(batch_average)
                        mouse['metrics'][key] = batch_average
                except Exception as details:
                    if self.verbosity > 0:
                        msg = '{} Encountered an exception.'
                        print(self._prefix_, msg.format(mouse['name']))
                        traceback.print_exception(details)
                finally:
                    mouse['candles'] = mouse['nn_gates'](inputs)[0]
                    mouse['candles'] = vstack(mouse['candles'].split(1))
                    mouse['candles'] = (mouse['candles'] * 3e3).clone()
        verbosity_check = [
            self.verbosity > 2,
            self.verbosity == 2 and study is False
            ]
        if any(verbosity_check):
            lr = mouse['warm_lr'].get_last_lr()[0]
            confidence = mouse['metrics']['confidence']
            acc = mouse['metrics']['acc']
            mae = mouse['metrics']['mae']
            mse = mouse['metrics']['mse']
            msg = f'{self._prefix_} {mouse["name"]} ' + '{}: {}'
            print(msg.format('Learning Rate', lr))
            print(msg.format('Confidence', confidence))
            print(msg.format('Accuracy', acc))
            print(msg.format('Mean Absolute Error', mae))
            print(msg.format('Mean Squared Error', mse))
            print(msg.format('Target', round(mouse['candles'][-1].item(), 2)))

    def research(self, symbol, candles, timeout=1):
        """Moirai research session, fully stocked with cheese and drinks."""
        if not all((
            len(candles.keys()) == self._n_features_,
            len(candles.index) >= self._min_size_
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
        target_accuracy = 97.0
        epochs = 0
        final_accuracy = 0
        volume_accuracy = 0
        while final_accuracy < target_accuracy:
            if epochs == timeout: break
            inputs = [[],[],[]]
            targets = [[],[],[]]
            batch_count = 0
            final_accuracy = 0
            volume_accuracy = 0
            for mouse in moirai:
                mouse['metrics']['confidence'] = 0
                mouse['metrics']['acc'] = 0
                mouse['metrics']['mae'] = 0
                mouse['metrics']['mse'] = 0
            for i in candles_index:
                if batch_count == batch:
                    for m_i, mouse in enumerate(moirai):
                        time_step(mouse, inputs[m_i], targets[m_i], study=True)
                    if self.verbosity > 2:
                        print('\n')
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
                    epochs=timestamps
                    )
                name = mouse['name']
                accuracy = mouse['metrics']['acc']
                if name == 'Lachesis':
                    volume_accuracy += accuracy
                else:
                    final_accuracy += accuracy
                if self.verbosity > 1:
                    print(prefix, 'Target:', mouse['data_targets'][-1], '\n')
            if epochs != 0:
                final_accuracy = float(final_accuracy)
                if final_accuracy != 0:
                    final_accuracy = final_accuracy / epochs
                volume_accuracy = float(volume_accuracy)
                if volume_accuracy != 0 != epochs:
                    volume_accuracy = volume_accuracy / epochs
            epochs += 1
            if self.verbosity > 0:
                msg = f'{prefix} ({epochs}) A moment of research '
                msg += 'yielded a price / volume accuracy of {}% / {}%'
                if self.verbosity > 1:
                    print('\n')
                print(msg.format(final_accuracy, volume_accuracy))
        sealed = [mouse['candles'].detach().cpu().numpy() for mouse in moirai]
        last_pred = float(sealed[0][-1].item())
        last_price = float(Atropos['data_final'][-1:, 3:4].item())
        proj_gain = percent_change(last_pred, last_price)
        self.predictions[symbol] = {
            'final_accuracy': round(final_accuracy, 2),
            'volume_accuracy': round(volume_accuracy, 2),
            'sealed_candles': sealed,
            'last_price': round(last_price, 2),
            'batch_pred': round(sealed[0][-1].item(), 5),
            'num_epochs': epochs,
            'metrics': dict(mouse['metrics']),
            'proj_gain': proj_gain,
            'proj_time': time.time()
            }
        if self.verbosity > 0:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
            print(f'{prefix} {symbol} Metrics;')
            for k, v in self.predictions[symbol].items():
                if k in ['final_accuracy', 'volume_accuracy']:
                    print(f'{prefix}     {k}: {v}%')
                elif k == 'sealed_candles':
                    if self.verbosity > 1:
                        print(f'{prefix}     {k}: {v}')
                elif k == 'metrics':
                    for m_k, m_v in v.items():
                        if m_k in ['acc', 'confidence']:
                            print(f'{prefix}     {m_k}: {m_v}%')
                        else:
                            print(f'{prefix}     {m_k}: {m_v}')
                else:
                    print(f'{prefix}     {k}: {v}')
            print('')
        data_labels = ['data_inputs', 'data_targets', 'data_final']
        for label in data_labels:
            Clotho[label] = None
            Lachesis[label] = None
            Atropos[label] = None
        self.__manage_state__(call_type=1)
        return True

    def forward(self, *inputs):
