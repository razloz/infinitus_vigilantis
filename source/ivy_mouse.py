"""Three blind mice to predict the future."""
import time
import torch
import torch.nn as nn
import traceback
import source.ivy_commons as icy
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import sqrt
from os.path import abspath
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class GRNN(nn.Module):
    """Gated Recurrent Neural Network"""
    def __init__(self, name, n_output, **kwargs):
        """Register modules and candles parameter."""
        super(GRNN, self).__init__()
        self._device_ = kwargs['device']
        self._n_hidden_ = int(kwargs['hidden_size'])
        self._n_output_ = int(n_output)
        self.__gru__ = nn.GRU(**kwargs)
        self.__batch_fn__ = nn.BatchNorm1d(self._n_hidden_)
        self.__linear_fn__ = nn.Linear(self._n_hidden_, self._n_output_)
        self.candles = dict()
        self.loss_fn = nn.HuberLoss(reduction='mean', delta=0.98)
        self.metrics = nn.ParameterDict()
        self.name = str(name)
        self.optimizer = RMSprop(
            self.__gru__.parameters(),
            lr=0.1,
            alpha=0.99,
            momentum=0.99,
            foreach=True,
            )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.97,
            patience=3,
            threshold=3e-4,
            threshold_mode='rel',
            cooldown=2,
            )
        self.to(self._device_)

    def forward(self, inputs):
        """Batch input to gated linear output."""
        inputs = self.__gru__(inputs)
        if type(inputs) is tuple: inputs = inputs[0]
        inputs = self.__batch_fn__(inputs)
        inputs = self.__linear_fn__(inputs)
        return inputs.clone()

    def give(self, cheese):
        """Sample cheese, create candles."""
        if type(cheese) is list:
            cheese = torch.hstack([*cheese])
        candles = self(cheese)
        return candles

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
        """Beckon the Moirai and Cauldron."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._labels_input_ = [
            'open', 'high', 'low', 'close',
            'price_dh', 'price_dl', 'price_mid'
            ]
        self._labels_loss_ = ['dist_close', 'dist_open', 'dist_wema']
        self._labels_pred_ = ['cdl_median', 'cdl_median', 'price_wema']
        self._labels_base_ = 'price_mid'
        self._labels_ = list(self._labels_input_)
        self._labels_ += self._labels_loss_
        self._labels_ += self._labels_pred_
        self._labels_.append(self._labels_base_)
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._state_path_ = abspath('./rnn/moirai.state')
        self.verbosity = int(verbosity)
        self._min_size_ = int(min_size)
        self._batch_size_ = self._min_size_
        self._n_features_ = int(len(self._labels_input_))
        self._n_hidden_ = int(self._n_features_ ** 2)
        self._tolerance_ = 0.01618033988749894
        p_gru = dict()
        p_gru['num_layers'] = self._min_size_
        p_gru['bias'] = True
        p_gru['batch_first'] = True
        p_gru['dropout'] = 0.34
        p_gru['device'] = self._device_
        p_cauldron = dict(p_gru)
        p_cauldron['input_size'] = 3
        p_cauldron['hidden_size'] = self._min_size_
        p_moira = dict(p_gru)
        p_moira['input_size'] = self._n_features_
        p_moira['hidden_size'] = self._n_hidden_
        # Awen the sentient cauldron.
        self.Awen = Cauldron('Awen', 3, **p_cauldron)
        # Atropos the mouse, sister of Clotho and Lachesis.
        self.Atropos = Moira('Atropos', 1, **p_moira)
        # Clotho the mouse, sister of Lachesis and Atropos.
        self.Clotho = Moira('Clotho', 1, **p_moira)
        # Lachesis the mouse, sister of Atropos and Clotho.
        self.Lachesis = Moira('Lachesis', 1, **p_moira)
        # Store predictions for sorting top picks
        self.predictions = nn.ParameterDict()
        self.to(self._device_)
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set batch_size to', self._batch_size_)
            print(self._prefix_, 'set min_size to', self._min_size_)
            print(self._prefix_, 'set n_features to', self._n_features_)
            print(self._prefix_, 'set n_hidden to', self._n_hidden_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'Awen params:')
            for _k, _v in p_cauldron.items():
                print(f'{_k}: {_v}')
            print(self._prefix_, 'Moira params:')
            for _k, _v in p_moira.items():
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

    def __time_step__(self, mouse, indices, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        vstack = torch.vstack
        if study is True:
            threshold = self._tolerance_
            cheese = mouse.candles['batch_inputs'][indices[0]:indices[1]]
            t_base = mouse.candles['targets_base'][indices[0]:indices[1]]
            t_loss = mouse.candles['targets_loss'][indices[0]:indices[1]]
            t_pred = mouse.candles['targets_pred'][indices[0]:indices[1]]
            t_base = vstack([t for t in t_base.split(1)])
            t_loss = vstack([t for t in t_loss.split(1)])
            t_pred = vstack([t for t in t_pred.split(1)])
            batch_size = cheese.shape[0]
            mouse.optimizer.zero_grad()
            candles = mouse.give(cheese)
            loss = mouse.loss_fn(candles.tanh(), t_loss.tanh())
            mouse.metrics['loss'] = loss.item()
            mouse.optimizer.step()
            mouse.scheduler.step(loss)
            candles = (t_base + (t_base * candles)).abs()
            mouse.candles['coated'] = candles.clone()
            difference = (candles - t_pred)
            correct = difference[difference >= -threshold]
            correct = correct[correct <= threshold].shape[0]
            mouse.metrics['acc'][0] += int(correct)
            mouse.metrics['acc'][1] += int(batch_size)
            if not mouse.metrics['mae']:
                mouse.metrics['mae'] = 0
            if not mouse.metrics['mse']:
                mouse.metrics['mse'] = 0
            mouse.metrics['mae'] += difference.abs().sum().item()
            mouse.metrics['mse'] += (difference ** 2).sum().item()
        else:
            with torch.no_grad():
                try:
                    mouse.metrics['mae'] = mouse.metrics['mae'] / epochs
                    mouse.metrics['mse'] = sqrt(mouse.metrics['mse'] / epochs)
                except Exception as details:
                    if self.verbosity > 0:
                        msg = '{} Encountered an exception.'
                        print(self._prefix_, msg.format(mouse.name))
                        traceback.print_exception(details)
                finally:
                    cheese = mouse.candles['batch_inputs'][indices[0]:]
                    t_base = mouse.candles['targets_base'][indices[0]:]
                    t_base = vstack([t for t in t_base.split(1)])
                    candles = mouse.give(cheese)
                    candles = (t_base + (t_base * candles)).abs()
                    mouse.candles['coated'] = candles.clone()
        if self.verbosity == 2 and study is False:
            lr = mouse.scheduler._last_lr
            msg = f'{self._prefix_} {mouse.name} ' + '{}: {}'
            print(msg.format('Learning Rate', lr))
            print(msg.format('Accuracy', mouse.metrics['acc']))
            print(msg.format('Loss', mouse.metrics['loss']))
            print(msg.format('Mean Absolute Error', mouse.metrics['mae']))
            print(msg.format('Mean Squared Error', mouse.metrics['mse']))
        elif self.verbosity > 2:
            print('candles:\n', candles)
            print('candles shape:\n', candles.shape)
            print('t_base:\n', t_base)
            print('t_base shape:\n', t_base.shape)
            if study is True:
                print('t_loss:\n', t_loss)
                print('t_loss shape:\n', t_loss.shape)
                print('t_pred:\n', t_pred)
                print('t_pred shape:\n', t_pred.shape)
            print('coated:\n', len(mouse.candles['coated']))

    def research(self, symbol, candles, timeout=9001, epoch_save=True):
        """Moirai research session, fully stocked with cheese and drinks."""
        _TK_ = icy.TimeKeeper()
        time_start = _TK_.reset
        candle_keys = candles.keys()
        if not all((
            len(candles.index) >= self._min_size_,
            *(l in candle_keys for l in self._labels_),
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
        base_targets = candles[self._labels_base_]
        moirai = [Atropos, Clotho, Lachesis]
        for m_i, mouse in enumerate(moirai):
            _mc = mouse.candles
            _inpt = candles[self._labels_input_]
            _loss = candles.pop(self._labels_loss_[m_i]).to_numpy()
            _pred = candles[self._labels_pred_[m_i]].to_numpy()
            _mc['batch_inputs'] = tensor(_inpt.to_numpy(), **p_tensor)
            _mc['batch_inputs'].requires_grad_(True)
            _mc['targets_base'] = tensor(base_targets.to_numpy(), **p_tensor)
            _mc['targets_loss'] = tensor(_loss[batch_size:], **p_tensor)
            _mc['targets_pred'] = tensor(_pred[batch_size:], **p_tensor)
            if self.verbosity > 1:
                _mn = mouse.name
                print(_mn, 'batch_inputs shape:', _mc['batch_inputs'].shape)
                print(_mn, 'targets_base shape:', _mc['targets_base'].shape)
                print(_mn, 'targets_loss shape:', _mc['targets_loss'].shape)
                print(_mn, 'targets_pred shape:', _mc['targets_pred'].shape)
        for _l in self._labels_pred_:
            if _l in candles.columns:
                candles.pop(_l)
        target_accuracy = 97.0
        target_loss = 1e-3
        target_mae = 10
        target_mse = 100
        epochs = 0
        final_accuracy = 0
        mouse_accuracy = 0
        Awen.reset_metrics()
        wicks, t_base, t_loss, t_pred = [[], [], [], []]
        while final_accuracy < target_accuracy:
            time_update = _TK_.update[0]
            if self.verbosity > 1:
                print(f'{prefix} epoch {epochs} elapsed time {time_update}.')
            break_condition = any((
                epochs == timeout,
                (Awen.metrics['loss'] is not None
                    and Awen.metrics['loss'] < target_loss),
                (Awen.metrics['mae'] is not None
                    and Awen.metrics['mae'] < target_mae),
                (Awen.metrics['mse'] is not None
                    and Awen.metrics['mse'] < target_mse),
                ))
            if break_condition:
                break
            else:
                epochs += 1
            final_accuracy = 0
            mouse_accuracy = 0
            Awen.reset_candles()
            for mouse in moirai:
                mouse.reset_candles()
                mouse.reset_metrics()
            last_batch = 0
            for i in batch_range:
                indices = (i, i + batch_size)
                if i < last_batch:
                    continue
                else:
                    last_batch = int(indices[1])
                if indices[1] <= batch_range[-1]:
                    for mouse in moirai:
                        time_step(mouse, indices, study=True)
                    Awen.optimizer.zero_grad()
                    wicks, t_base, t_loss, t_pred = [[], [], [], []]
                    ii, iii = indices
                    for mouse in moirai:
                        wicks.append(mouse.candles['coated'])
                        t_base.append(mouse.candles['targets_base'][ii:iii])
                        t_loss.append(mouse.candles['targets_loss'][ii:iii])
                        t_pred.append(mouse.candles['targets_pred'][ii:iii])
                    wicks = hstack(wicks)
                    t_base = vstack(t_base).H
                    t_loss = vstack(t_loss).H
                    t_pred = vstack(t_pred).H
                    coated_wicks = Awen.give(wicks)
                    loss = Awen.loss_fn(coated_wicks.tanh(), t_loss.tanh())
                    loss.backward()
                    Awen.optimizer.step()
                    Awen.scheduler.step(loss)
                    threshold = float(t_pred.max().item() * tolerance)
                    coated_candles = (t_base + (t_base * coated_wicks)).abs()
                    Awen.candles['coated'] = coated_candles.clone()
                    Awen.candles['sealed'].append(coated_candles.clone())
                    difference = (coated_candles - t_pred)
                    correct = difference[difference >= -threshold]
                    correct = correct[correct <= threshold].shape[0]
                    cauldron_size = difference.shape[0] * difference.shape[1]
                    Awen.metrics['acc'][0] += int(correct)
                    Awen.metrics['acc'][1] += int(cauldron_size)
                    Awen.metrics['loss'] = loss.item()
                    if not Awen.metrics['mae']:
                        Awen.metrics['mae'] = 0
                    if not Awen.metrics['mse']:
                        Awen.metrics['mse'] = 0
                    Awen.metrics['mae'] += difference.abs().sum().item()
                    Awen.metrics['mse'] += (difference ** 2).sum().item()
                    if self.verbosity > 2:
                        for m_k, m_v in Awen.metrics.items():
                            print(prefix, f'Awen {m_k}: {m_v}')
                else:
                    for mouse in moirai:
                        time_step(mouse, (-batch_size, None), epochs=timestamps)
                    wicks = [m.candles['coated'] for m in moirai]
                    t_key = 'targets_base'
                    t_base = [m.candles[t_key][-batch_size:] for m in moirai]
                    t_base = vstack(t_base).H
                    coated_wicks = Awen.give(wicks)
                    coated_candles = (t_base + (t_base * coated_wicks)).abs()
                    Awen.candles['coated'] = coated_candles.clone()
                    seal = vstack(Awen.candles['sealed'])
                    Awen.candles['sealed'] = vstack([seal, coated_candles])
                    Awen.metrics['mae'] = Awen.metrics['mae'] / epochs
                    Awen.metrics['mse'] = Awen.metrics['mse'] / epochs
                    n_correct = Awen.metrics['acc'][0]
                    n_total = Awen.metrics['acc'][1]
                    n_wrong = n_total - n_correct
                    final_accuracy = 100 * abs((n_wrong - n_total) / n_total)
                    final_accuracy = round(final_accuracy, 3)
                    n_correct = 0
                    n_total = 0
                    for mouse in moirai:
                        n_correct += mouse.metrics['acc'][0]
                        n_total += mouse.metrics['acc'][1]
                    n_wrong = n_total - n_correct
                    mouse_accuracy = 100 * abs((n_wrong - n_total) / n_total)
                    mouse_accuracy = round(mouse_accuracy, 3)
                    break
            if epoch_save:
                self.__manage_state__(call_type=1)
            if self.verbosity > 0:
                print(f'{prefix} ({epochs}) A moment of research yielded;')
                print(f'    a cauldron accuracy of {final_accuracy}%')
                print(f'    a mouse accuracy of {mouse_accuracy}%')
                if self.verbosity > 1:
                    print(f'coated_candles: {coated_candles.shape}')
                    print(coated_candles)
                    print('metrics:')
                    for metric_k, metric_v in Awen.metrics.items():
                        print(f'    {metric_k}:', metric_v)
                    print('')
        last_pred = float(Awen.candles['coated'][-1][0].item())
        last_price = float(Atropos.candles['targets_pred'][-1])
        proj_gain = float(((last_pred - last_price) / last_price) * 100)
        timestamp = time.time()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        self.predictions[symbol] = {
            'cauldron loss': float(Awen.metrics['loss']),
            'cauldron_accuracy': final_accuracy,
            'mouse_accuracy': mouse_accuracy,
            'coated_candles': Awen.candles['coated'].detach().cpu().numpy(),
            'sealed_candles': Awen.candles['sealed'].detach().cpu().numpy(),
            'last_price': round(last_price, 3),
            'batch_pred': round(last_pred, 3),
            'num_epochs': epochs,
            'metrics': dict(Awen.metrics),
            'proj_gain': proj_gain,
            'proj_timestamp': timestamp,
            'proj_time_str': time_str,
            }
        if self.verbosity > 0:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
            print(f'{prefix} {symbol} Metrics;')
            for k, v in self.predictions[symbol].items():
                if k in ['cauldron_accuracy', 'mouse_accuracy']:
                    print(f'{prefix}     {k}: {v}%')
                elif k in ['sealed_candles', 'coated_candles']:
                    if self.verbosity > 1:
                        print(f'{prefix}     {k}: {v}')
                        print(f'{prefix}     {k} shape: {v.shape}')
                elif k == 'metrics':
                    for m_k, m_v in v.items():
                        print(f'{prefix}     {m_k}: {m_v}')
                else:
                    print(f'{prefix}     {k}: {v}')
            time_elapsed = _TK_.final[0]
            print(f'{prefix} final elapsed time {time_elapsed}.')
            print('')
        Awen.release_tensors()
        for mouse in moirai:
            mouse.release_tensors()
        self.__manage_state__(call_type=1)
        return True
