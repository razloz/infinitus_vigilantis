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
    def __init__(self, name, batch_size, *args, **kwargs):
        """Register modules and candles parameter."""
        super(GRNN, self).__init__()
        self._device_ = kwargs['device']
        self.__gru__ = nn.GRU(**kwargs)
        self.__batch_fn__ = nn.BatchNorm1d(int(kwargs['hidden_size']))
        self.__linear_fn__ = nn.Linear(int(kwargs['hidden_size']), 1)
        self.candles = dict()
        self.loss_fn = nn.HuberLoss(reduction='mean', delta=0.97)
        self.metrics = nn.ParameterDict()
        self.name = str(name)
        self.optimizer = RMSprop(self.__gru__.parameters(), lr=3e-3)
        self.scheduler = WarmRestarts(self.optimizer, int(batch_size * 0.5))
        self.to(self._device_)

    def forward(self, inputs):
        """Batch input to gated linear output."""
        inputs = self.__gru__(inputs)
        if type(inputs) is tuple:
            inputs = inputs[0]
        inputs = inputs.tanh().log_softmax(0)
        inputs = self.__batch_fn__(inputs)
        inputs = self.__linear_fn__(inputs)
        return inputs

    def give(self, cheese, wax):
        """Sample cheese, create candles."""
        if type(cheese) is list:
            cheese = torch.hstack(cheese)
        wicks = torch.exp(self(cheese))
        candles = (wax * wicks).relu()
        return candles.clone()

    def release_tensors(self):
        """Remove batch and target tensors from memory."""
        self.candles['inputs'] = None
        self.candles['targets'] = None
        self.candles['wax'] = None

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
    def __init__(self, batch_size, verbosity=0, min_size=90, *args, **kwargs):
        """Beckon the Norns."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._features_ = [
            'open', 'high', 'low', 'close', 'volume', 'num_trades',
            'vol_wma_price', 'trend', 'fib_retrace_0.236', 'fib_retrace_0.382',
            'fib_retrace_0.5', 'fib_retrace_0.618', 'fib_retrace_0.786',
            'fib_retrace_0.886', 'price_zs', 'price_sdev', 'price_wema',
            'price_dh', 'price_dl', 'price_mid', 'volume_zs', 'volume_sdev',
            'volume_wema', 'volume_dh', 'volume_dl', 'volume_mid', 'median_oc',
            'median_hl', 'wema_dist_hl', 'wema_dist_vp', 'wema_dist_oc',
            'wema_dist_v', 'cdl_change'
            ]
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._state_path_ = abspath('./rnn')
        self.verbosity = int(verbosity)
        self._min_size_ = int(min_size)
        self._batch_size_ = int(batch_size)
        self._n_features_ = len(self._features_) - 1
        self._n_hidden_ = int(self._n_features_ * 2)
        self._n_layers_ = int(self._n_features_ ** 2)
        self._tolerance_ = 0.01618033988749894
        p_gru = dict()
        p_gru['input_size'] = self._n_features_
        p_gru['hidden_size'] = self._n_hidden_
        p_gru['num_layers'] = self._n_layers_
        p_gru['bias'] = True
        p_gru['batch_first'] = True
        p_gru['dropout'] = 0.34
        p_gru['device'] = self._device_
        # Awen the sentient cauldron.
        self.Awen = Cauldron('Awen', self._batch_size_, **p_gru)
        # Atropos the mouse, sister of Clotho and Lachesis.
        self.Atropos = Moira('Atropos', self._batch_size_, **p_gru)
        # Clotho the mouse, sister of Lachesis and Atropos.
        self.Clotho = Moira('Clotho', self._batch_size_, **p_gru)
        # Lachesis the mouse, sister of Atropos and Clotho.
        self.Lachesis = Moira('Lachesis', self._batch_size_, **p_gru)
        # Store predictions for sorting top picks
        self.predictions = nn.ParameterDict({'num_epochs': 0})
        self.to(self._device_)
        if self.verbosity > 0:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set batch_size to', self._batch_size_)
            print(self._prefix_, 'set min_size to', self._min_size_)
            print(self._prefix_, 'set n_features to', self._n_features_)
            print(self._prefix_, 'set n_hidden to', self._n_hidden_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'GRU params:')
            for _k, _v in p_gru.items():
                print(f'    {_k}: {_v}')

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            state_path = f'{self._state_path_}/moirai.state'
            if call_type == 0:
                state = torch.load(state_path, map_location=self._device_type_)
                self.load_state_dict(state['moirai'])
                for key, value in state['predictions'].items():
                    self.predictions[key] = value
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            elif call_type == 1:
                moirai = self.state_dict()
                predictions = dict()
                for key, value in self.predictions.items():
                    if key in ['coated_candles', 'sealed_candles']:
                        continue
                    predictions[key] = value
                torch.save(
                    dict(moirai=moirai, predictions=predictions),
                    state_path,
                    )
                if self.verbosity > 2:
                    print(self._prefix_, 'Saved RNN state.')
        except Exception as details:
            if self.verbosity > 2:
                print(self._prefix_, 'Encountered an exception.')
                traceback.print_exception(details)

    def __time_step__(self, norn, indices, mode, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        vstack = torch.vstack
        batch_start, batch_stop = indices
        name = norn.name
        wax = norn.candles['wax']
        if mode == 'train':
            self.train()
        elif mode == 'eval':
            self.eval()
        if study is True:
            threshold = wax * self._tolerance_
            cheese = norn.candles['inputs'][batch_start:batch_stop]
            targets = norn.candles['targets'][batch_start:batch_stop]
            targets = vstack([t for t in targets.split(1)])
            batch_size = cheese.shape[0]
            norn.optimizer.zero_grad()
            candles = norn.give(cheese, wax)
            loss = norn.loss_fn(candles, targets)
            loss.backward()
            difference = (candles - targets)
            correct = difference[difference >= -threshold]
            correct = correct[correct <= threshold].shape[0]
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
                    candles = norn.give(cheese, wax)
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
            print('coated:\n', len(mouse.candles['coated']))

    def research(self, symbol, candles, mode, timeout=1, epoch_save=False):
        """Moirai research session, fully stocked with cheese and drinks."""
        _TK_ = icy.TimeKeeper()
        time_start = _TK_.reset
        candle_keys = candles.keys()
        if not all((
            len(candles.index) >= self._min_size_,
            *(key in candle_keys for key in self._features_),
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
        norns = [Awen, Atropos, Clotho, Lachesis]
        _target_keys = ['volume', 'close', 'open', 'price_wema']
        for t_index, norn in enumerate(norns):
            norn.reset_candles()
            norn.reset_metrics()
            _candles = norn.candles
            _name = norn.name
            _target = _target_keys[t_index]
            _features = [l for l in self._features_ if l != _target]
            _inputs = candles[_features].to_numpy()
            _targets = candles[_target].to_numpy()[batch_size:]
            _candles['inputs'] = tensor(_inputs, **p_tensor)
            _candles['inputs'].requires_grad_(True)
            _candles['targets'] = tensor(_targets, **p_tensor)
            _candles['wax'] = round(_candles['targets'].mean().item(), 3)
            if self.verbosity > 1:
                print(norn.name, 'inputs:', _candles['inputs'].shape)
                print(norn.name, 'targets:', _candles['targets'].shape)
                print(norn.name, 'wax:', _candles['wax'])
        target_accuracy = 89.0
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
                        time_step(norn, indices, mode, study=True)
                else:
                    n_total = 0
                    n_correct = 0
                    indices = (-batch_size, None)
                    for norn in norns:
                        time_step(norn, indices, mode, epochs=timestamps)
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
        last_pred = float(Atropos.candles['coated'][-1].item())
        last_price = float(candles['close'][-1])
        proj_gain = float(((last_pred - last_price) / last_price) * 100)
        coated, sealed = list(), list()
        candle_mae, candle_mse, candle_loss = 0, 0, 0
        volume_mae, volume_mse, volume_loss = 0, 0, 0
        for norn in norns:
            coated.append(norn.candles['coated'])
            sealed.append(vstack(norn.candles['sealed']))
            if norn.name == 'Awen':
                volume_mae += norn.metrics['mae']
                volume_mse += norn.metrics['mse']
                volume_loss += norn.metrics['loss']
            else:
                candle_mae += norn.metrics['mae']
                candle_mse += norn.metrics['mse']
                candle_loss += norn.metrics['loss']
        total_mae = (round(candle_mae, 5), round(volume_mae, 5))
        total_mse = (round(candle_mse, 5), round(volume_mse, 5))
        total_loss = (round(candle_loss, 5), round(volume_loss, 5))
        coated = hstack(coated)
        sealed = hstack(sealed)
        timestamp = time.time()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        self.predictions['symbol'] = symbol.upper()
        self.predictions['num_epochs'] += epochs
        self.predictions['final_accuracy'] = final_accuracy
        self.predictions['total_mae'] = total_mae
        self.predictions['total_mse'] = total_mse
        self.predictions['total_loss'] = total_loss
        self.predictions['last_price'] = round(last_price, 3)
        self.predictions['batch_pred'] = round(last_pred, 3)
        self.predictions['proj_gain'] = round(proj_gain, 3)
        self.predictions['proj_timestamp'] = timestamp
        self.predictions['proj_time_str'] = time_str
        self.predictions['coated_candles'] = coated.detach().cpu().numpy()
        self.predictions['sealed_candles'] = sealed.detach().cpu().numpy()
        for norn in norns:
            norn.release_tensors()
        self.__manage_state__(call_type=1)
        if self.verbosity > 0:
            if self.verbosity > 1:
                for k, v in self.predictions.items():
                    if k in ['coated_candles', 'sealed_candles']:
                        continue
                    print(f'{k}: {v}')
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
            time_elapsed = _TK_.final[0]
            print(f'{prefix} final elapsed time {time_elapsed}.')
            print('')
        return True
