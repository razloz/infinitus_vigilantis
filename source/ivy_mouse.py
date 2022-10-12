"""Three blind mice to predict the future."""
import time
import torch
import torch.nn as nn
import traceback
import source.ivy_commons as icy
from torch.optim import RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as WarmRestarts
from math import sqrt
from os.path import abspath
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class GRNN(nn.Module):
    """Gated Recurrent Neural Network"""
    def __init__(self, name, **kwargs):
        """Register modules and candles parameter."""
        super(GRNN, self).__init__()
        self._device_ = kwargs['device']
        self._n_hidden_ = int(kwargs['hidden_size'])
        self._n_output_ = 2
        self.__gru__ = nn.GRU(**kwargs)
        self.__batch_fn__ = nn.BatchNorm1d(self._n_hidden_)
        self.__linear_fn__ = nn.Linear(self._n_hidden_, self._n_output_)
        self.candles = dict()
        self.loss_fn = nn.HuberLoss(reduction='mean', delta=0.97)
        self.metrics = nn.ParameterDict()
        self.name = str(name)
        self.optimizer = RMSprop(
            self.__gru__.parameters(),
            lr=0.03,
            alpha=0.97,
            momentum=0.97,
            )
        self.scheduler = WarmRestarts(self.optimizer, 17)
        self.to(self._device_)

    def forward(self, inputs):
        """Batch input to gated linear output."""
        inputs = self.__gru__(inputs)
        if type(inputs) is tuple: inputs = inputs[0]
        inputs = self.__batch_fn__(inputs)
        inputs = self.__linear_fn__(inputs)
        return inputs

    def give(self, cheese):
        """Sample cheese, create candles."""
        if type(cheese) is list:
            cheese = torch.hstack([*cheese])
        _max = cheese.max()
        _min = cheese.min()
        wicks = (_max - ((_max - _min) * 0.5))
        wax = (1 + self(cheese))
        candles = (wicks * wax).relu()
        return candles.clone()

    def release_tensors(self):
        """Remove batch and target tensors from memory."""
        self.candles['batch_inputs'] = None
        self.candles['targets_base'] = None
        self.candles['targets_loss'] = None
        self.candles['targets_pred'] = None

    def reset_candles(self):
        """Clear previous candles."""
        self.candles['coated'] = None
        self.candles['sealed'] = list()

    def reset_metrics(self):
        """Clear previous metrics."""
        self.metrics['acc'] = [0, 0]
        self.metrics['loss'] = None
        self.metrics['mae'] = None
        self.metrics['mse'] = None


class Cauldron(GRNN):
    """A wax encrusted cauldron sits before you, bubbling occasionally."""
    def __init__(self, *args, **kwargs):
        """Assists in the creation of candles."""
        super(Cauldron, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """*bubble**bubble* *bubble*"""
        return super(Cauldron, self).forward(*args, **kwargs)


class Moira(GRNN):
    """A clever, cheese candle constructing, future predicting mouse."""
    def __init__(self, *args, **kwargs):
        """The number of the counting shall be three."""
        super(Moira, self).__init__(*args, **kwargs)


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, verbosity=0, min_size=90, *args, **kwargs):
        """Beckon the Norns."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._labels_ = {
            'Cheese': [
                'open', 'high', 'low', 'close',
                'price_dh', 'price_dl', 'price_mid',
                'cdl_median', 'price_wema'
                ],
            'Clotho': [
                'price_wema', 'cdl_median'
                ],
            'Lachesis': [
                'high', 'low'
                ],
            'Atropos': [
                'open', 'close'
                ],
            'Awen': [
                'price_dh', 'price_dl'
                ],
            }
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._state_path_ = abspath('./rnn/moirai.state')
        self.verbosity = int(verbosity)
        self._min_size_ = int(min_size)
        self._batch_size_ = self._min_size_
        self._n_features_ = int(len(self._labels_['Cheese']) - 2)
        self._n_hidden_ = int(self._n_features_ ** 2)
        self._tolerance_ = 0.01618033988749894
        p_gru = dict()
        p_gru['input_size'] = self._n_features_
        p_gru['hidden_size'] = self._n_hidden_
        p_gru['num_layers'] = self._n_hidden_
        p_gru['bias'] = True
        p_gru['batch_first'] = True
        p_gru['dropout'] = 0.13
        p_gru['device'] = self._device_
        # Awen the sentient cauldron.
        self.Awen = Cauldron('Awen', **p_gru)
        # Atropos the mouse, sister of Clotho and Lachesis.
        self.Atropos = Moira('Atropos', **p_gru)
        # Clotho the mouse, sister of Lachesis and Atropos.
        self.Clotho = Moira('Clotho', **p_gru)
        # Lachesis the mouse, sister of Atropos and Clotho.
        self.Lachesis = Moira('Lachesis', **p_gru)
        # Store predictions for sorting top picks
        self.predictions = nn.ParameterDict()
        self.targets_zero = torch.zeros(self._batch_size_, 2, **self._p_tensor_)
        self.to(self._device_)
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set batch_size to', self._batch_size_)
            print(self._prefix_, 'set min_size to', self._min_size_)
            print(self._prefix_, 'set n_features to', self._n_features_)
            print(self._prefix_, 'set n_hidden to', self._n_hidden_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'GRU params:')
            for _k, _v in p_gru.items():
                print(f'{_k}: {_v}')

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

    def __time_step__(self, norn, indices, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        vstack = torch.vstack
        batch_start, batch_stop = indices
        if study is True:
            threshold = self._tolerance_
            cheese = norn.candles['inputs'][batch_start:batch_stop]
            targets = norn.candles['targets'][batch_start:batch_stop]
            targets = vstack([t for t in targets.split(1)])
            batch_size = cheese.shape[0]
            norn.optimizer.zero_grad()
            candles = norn.give(cheese)
            difference = (candles - targets).tanh()
            correct = difference[difference >= -threshold]
            correct = correct[correct <= threshold].shape[0]
            loss = norn.loss_fn(difference, self.targets_zero)
            loss.backward()
            norn.metrics['loss'] = loss.item()
            norn.candles['coated'] = candles.clone()
            norn.candles['sealed'].append(candles.clone())
            norn.metrics['acc'][0] += int(correct)
            norn.metrics['acc'][1] += int(batch_size)
            if not norn.metrics['mae']:
                norn.metrics['mae'] = 0
            if not norn.metrics['mse']:
                norn.metrics['mse'] = 0
            norn.metrics['mae'] += difference.abs().sum().item()
            norn.metrics['mse'] += (difference ** 2).sum().item()
        else:
            with torch.no_grad():
                try:
                    norn.metrics['mae'] = norn.metrics['mae'] / epochs
                    norn.metrics['mse'] = sqrt(norn.metrics['mse'] / epochs)
                except Exception as details:
                    if self.verbosity > 0:
                        msg = '{} Encountered an exception.'
                        print(self._prefix_, msg.format(norn.name))
                        traceback.print_exception(details)
                finally:
                    cheese = norn.candles['inputs'][batch_start:]
                    candles = norn.give(cheese)
                    norn.candles['coated'] = candles.clone()
                    norn.candles['sealed'].append(candles.clone())
        if self.verbosity == 2 and study is False:
            msg = f'{self._prefix_} {norn.name} ' + '{}: {}'
            lr = norn.scheduler._last_lr
            print(msg.format('Learning Rate', lr))
            print(msg.format('Accuracy', norn.metrics['acc']))
            print(msg.format('Loss', norn.metrics['loss']))
            print(msg.format('Mean Absolute Error', norn.metrics['mae']))
            print(msg.format('Mean Squared Error', norn.metrics['mse']))
        elif self.verbosity > 2:
            msg = f'{self._prefix_} {norn.name} ' + '{}: {}'
            lr = norn.scheduler._last_lr
            print(msg.format('Learning Rate', lr))
            print(msg.format('candles:\n', candles))
            print(msg.format('candles shape:\n', candles.shape))
            if study is True:
                print(msg.format('targets:\n', targets))
                print(msg.format('targets shape:\n', targets.shape))
            print('coated:\n', len(norn.candles['coated']))

    def research(self, symbol, candles, timeout=34, epoch_save=True):
        """Moirai research session, fully stocked with cheese and drinks."""
        _TK_ = icy.TimeKeeper()
        time_start = _TK_.reset
        candle_keys = candles.keys()
        if not all((
            len(candles.index) >= self._min_size_,
            *(l in candle_keys for l in self._labels_['Cheese']),
            )): return False
        self.__manage_state__(call_type=0)
        Awen = self.Awen
        Atropos = self.Atropos
        Clotho = self.Clotho
        Lachesis = self.Lachesis
        time_step = self.__time_step__
        batch_size = self._batch_size_
        prefix = self._prefix_
        p_tensor = self._p_tensor_
        tolerance = self._tolerance_
        hstack = torch.hstack
        vstack = torch.vstack
        tensor = torch.tensor
        candles = candles[self._min_size_:]
        batch_index = 0
        batch_fit = 0
        for n_batch in range(len(candles.index)):
            batch_index += 1
            if batch_index % batch_size == 0:
                batch_fit += batch_size
                batch_index = 0
        candles = candles[-batch_fit:]
        timestamps = len(candles.index)
        batch_range = range(timestamps - batch_size)
        norns = [Atropos, Clotho, Lachesis, Awen]
        for norn in norns:
            norn.reset_candles()
            norn.reset_metrics()
            _candles = norn.candles
            _targets = self._labels_[norn.name]
            _labels = self._labels_['Cheese']
            _labels = [l for l in _labels if not l in _targets]
            _input = candles[_labels].to_numpy()
            _target = candles[_targets].to_numpy()
            _target = _target[batch_size:]
            _candles['inputs'] = tensor(_input, **p_tensor)
            _candles['inputs'].requires_grad_(True)
            _candles['targets'] = tensor(_target, **p_tensor)
            if self.verbosity > 1:
                _name = norn.name
                print(_name, 'inputs shape:',
                      _candles['inputs'].shape)
                print(_name, 'targets shape:',
                      _candles['targets'].shape)
        target_accuracy = 55.0
        target_loss = 0.1 / batch_size
        target_mae = 1e-3
        target_mse = target_mae ** 2
        if self.verbosity > 1:
            print(prefix, 'target_accuracy', target_accuracy)
            print(prefix, 'target_loss', target_loss)
            print(prefix, 'target_mae', target_mae)
            print(prefix, 'target_mse', target_mse)
        epochs = 0
        final_accuracy = 0
        mouse_accuracy = 0
        Awen.reset_metrics()
        wicks, targets = [[], []]
        while final_accuracy < target_accuracy:
            time_update = _TK_.update[0]
            if self.verbosity > 1:
                print(f'{prefix} epoch {epochs} elapsed time {time_update}.')
            for norn in norns:
                break_condition = all((
                    (norn.metrics['loss'] is not None
                        and norn.metrics['loss'] < target_loss),
                    (norn.metrics['mae'] is not None
                        and norn.metrics['mae'] < target_mae),
                    (norn.metrics['mse'] is not None
                        and norn.metrics['mse'] < target_mse),
                    ))
                if break_condition:
                    break
            if epochs == timeout or break_condition:
                break
            else:
                epochs += 1
            for norn in norns:
                norn.reset_candles()
                norn.reset_metrics()
            final_accuracy = 0
            last_batch = 0
            for i in batch_range:
                indices = (i, i + batch_size)
                if i < last_batch:
                    continue
                else:
                    last_batch = int(indices[1])
                if last_batch <= batch_range[-1]:
                    for norn in norns:
                        time_step(norn, indices, study=True)
                else:
                    n_total = 0
                    n_correct = 0
                    for norn in norns:
                        time_step(norn, (-batch_size, None), epochs=timestamps)
                        n_correct += norn.metrics['acc'][0]
                        n_total += norn.metrics['acc'][1]
                    n_wrong = n_total - n_correct
                    final_accuracy = 100 * abs((n_wrong - n_total) / n_total)
                    final_accuracy = round(final_accuracy, 3)
                    break
            for norn in norns:
                norn.optimizer.step()
                norn.scheduler.step()
            if epoch_save:
                self.__manage_state__(call_type=1)
            if self.verbosity > 0:
                msg = f'({epochs}) A moment of research yielded '
                msg += f'a final accuracy of {final_accuracy}%'
                print(prefix, msg)
        last_pred = float(Clotho.candles['coated'][-1][1].item())
        last_price = float(candles['close'][-1])
        proj_gain = float(((last_pred - last_price) / last_price) * 100)
        coated = list()
        sealed = list()
        for norn in norns:
            coated.append(norn.candles['coated'])
            sealed.append(vstack(norn.candles['sealed']))
        coated = hstack(coated)
        sealed = hstack(sealed)
        timestamp = time.time()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        self.predictions[symbol] = {
            'final_accuracy': final_accuracy,
            'coated_candles': coated.detach().cpu().numpy(),
            'sealed_candles': sealed.detach().cpu().numpy(),
            'last_price': round(last_price, 3),
            'batch_pred': round(last_pred, 3),
            'num_epochs': epochs,
            'metrics': {norn.name: dict(norn.metrics) for norn in norns},
            'proj_gain': proj_gain,
            'proj_timestamp': timestamp,
            'proj_time_str': time_str,
            }
        for norn in norns:
            norn.release_tensors()
        self.__manage_state__(call_type=1)
        if self.verbosity > 0:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
            time_elapsed = _TK_.final[0]
            print(f'{prefix} final elapsed time {time_elapsed}.')
            print('')
        return True
